package controller

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/pkoukk/tiktoken-go"
	_ "golang.org/x/image/webp"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"net/http"
	"one-api/common"
	"one-api/model"
	"strconv"
	"strings"
)

var stopFinishReason = "stop"

// tokenEncoderMap won't grow after initialization
var tokenEncoderMap = map[string]*tiktoken.Tiktoken{}
var defaultTokenEncoder *tiktoken.Tiktoken

func InitTokenEncoders() {
	common.SysLog("initializing token encoders")
	gpt35TokenEncoder, err := tiktoken.EncodingForModel("gpt-3.5-turbo")
	if err != nil {
		common.FatalLog(fmt.Sprintf("failed to get gpt-3.5-turbo token encoder: %s", err.Error()))
	}
	defaultTokenEncoder = gpt35TokenEncoder
	gpt4TokenEncoder, err := tiktoken.EncodingForModel("gpt-4")
	if err != nil {
		common.FatalLog(fmt.Sprintf("failed to get gpt-4 token encoder: %s", err.Error()))
	}
	for m := range common.ModelRatio {
		if strings.HasPrefix(m, "gpt-3.5") {
			tokenEncoderMap[m] = gpt35TokenEncoder
		} else if strings.HasPrefix(m, "gpt-4") {
			tokenEncoderMap[m] = gpt4TokenEncoder
		} else {
			tokenEncoderMap[m] = nil
		}
	}
	common.SysLog("token encoders initialized")
}

func getTokenEncoder(model string) *tiktoken.Tiktoken {
	tokenEncoder, ok := tokenEncoderMap[model]
	if ok && tokenEncoder != nil {
		return tokenEncoder
	}
	if ok {
		tokenEncoder, err := tiktoken.EncodingForModel(model)
		if err != nil {
			common.SysError(fmt.Sprintf("failed to get token encoder for model %s: %s, using encoder for gpt-3.5-turbo", model, err.Error()))
			tokenEncoder = defaultTokenEncoder
		}
		tokenEncoderMap[model] = tokenEncoder
		return tokenEncoder
	}
	return defaultTokenEncoder
}

func getTokenNum(tokenEncoder *tiktoken.Tiktoken, text string) int {
	if common.ApproximateTokenEnabled {
		return int(float64(len(text)) * 0.38)
	}
	return len(tokenEncoder.Encode(text, nil, nil))
}

func resolveResolution(originW, originH, maxLongSide, maxShortSide int) (int, int) {
	w, h := float64(originW), float64(originH)
	ratio := w / h

	if w > h {
		if w > float64(maxLongSide) {
			w = float64(maxLongSide)
			h = w / ratio
		}
		if h > float64(maxShortSide) {
			h = float64(maxShortSide)
			w = h * ratio
		}
	} else {
		if h > float64(maxLongSide) {
			h = float64(maxLongSide)
			w = h * ratio
		}
		if w > float64(maxShortSide) {
			w = float64(maxShortSide)
			h = w / ratio
		}
	}

	return int(math.Floor(w)), int(math.Floor(h))
}

func countTokenImage(img *ContentPartImageUrl) (int, error) {
	if img.Detail == "low" {
		return 85, nil
	}

	var buf []byte
	if strings.HasPrefix(img.Url, "data:image/") {
		splitData := strings.Split(img.Url, ",")
		if len(splitData) != 2 {
			return 0, fmt.Errorf("invalid image data url")
		}
		var err error
		buf, err = base64.StdEncoding.DecodeString(splitData[1])
		if err != nil {
			return 0, err
		}
	} else {
		resp, err := http.Get(img.Url)
		if err != nil {
			return 0, err
		}
		buf, err = io.ReadAll(resp.Body)
		if err != nil {
			return 0, err
		}
		err = resp.Body.Close()
		if err != nil {
			return 0, err
		}
	}

	// get image width & height
	i, _, err := image.Decode(bytes.NewReader(buf))
	if err != nil {
		return 0, err
	}

	bounds := i.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y

	width, height = resolveResolution(width, height, 2000, 768)

	h := math.Ceil(float64(height) / 512)
	w := math.Ceil(float64(width) / 512)
	tokens := 85 + int(h*w)*170
	return tokens, nil
}

func countTokenImages(images []*ContentPartImageUrl) (int, []error) {
	tokens := 0
	var errs []error
	for _, img := range images {
		if token, err := countTokenImage(img); err != nil {
			errs = append(errs, err)
			tokens += 765
		} else {
			tokens += token
		}
	}
	return tokens, errs
}

func countTokenMessages(messages []Message, model string) int {
	tokenEncoder := getTokenEncoder(model)
	// Reference:
	// https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
	// https://github.com/pkoukk/tiktoken-go/issues/6
	//
	// Every message follows <|start|>{role/name}\n{content}<|end|>\n
	var tokensPerMessage int
	var tokensPerName int
	if model == "gpt-3.5-turbo-0301" {
		tokensPerMessage = 4
		tokensPerName = -1 // If there's a name, the role is omitted
	} else {
		tokensPerMessage = 3
		tokensPerName = 1
	}
	tokenNum := 0
	for _, message := range messages {
		tokenNum += tokensPerMessage
		tokenNum += getTokenNum(tokenEncoder, message.Content)
		tokenNum += getTokenNum(tokenEncoder, message.Role)
		if message.Name != nil {
			tokenNum += tokensPerName
			tokenNum += getTokenNum(tokenEncoder, *message.Name)
		}
	}
	tokenNum += 3 // Every reply is primed with <|start|>assistant<|message|>
	return tokenNum
}

func countTokenInput(input any, model string) int {
	switch input.(type) {
	case string:
		return countTokenText(input.(string), model)
	case []string:
		text := ""
		for _, s := range input.([]string) {
			text += s
		}
		return countTokenText(text, model)
	}
	return 0
}

func countTokenText(text string, model string) int {
	tokenEncoder := getTokenEncoder(model)
	return getTokenNum(tokenEncoder, text)
}

func reformatJson(v json.RawMessage, indent bool) []byte {
	var data any
	if err := json.Unmarshal(v, &data); err != nil {
		return v
	}
	var buf []byte
	var err error
	if indent {
		buf, err = json.MarshalIndent(data, "", "  ")
	} else {
		buf, err = json.Marshal(data)
	}
	if err != nil {
		return v
	}
	return buf
}

func countTokenFunctions(functions json.RawMessage, functionCall json.RawMessage, model string) int {
	if functions == nil {
		return 0
	}
	tokenEncoder := getTokenEncoder(model)

	tokens := getTokenNum(tokenEncoder, string(reformatJson(functions, true)))
	tokens = int(float64(tokens) * 0.6)

	tokens += getTokenNum(tokenEncoder, string(reformatJson(functionCall, false)))

	return tokens
}

func errorWrapper(err error, code string, statusCode int) *OpenAIErrorWithStatusCode {
	openAIError := OpenAIError{
		Message: err.Error(),
		Type:    "one_api_error",
		Code:    code,
	}
	return &OpenAIErrorWithStatusCode{
		OpenAIError: openAIError,
		StatusCode:  statusCode,
	}
}

func shouldDisableChannel(err *OpenAIError, statusCode int) bool {
	if !common.AutomaticDisableChannelEnabled {
		return false
	}
	if err == nil {
		return false
	}
	if statusCode == http.StatusUnauthorized {
		return true
	}
	if err.Type == "insufficient_quota" || err.Code == "invalid_api_key" || err.Code == "account_deactivated" {
		return true
	}
	return false
}

func setEventStreamHeaders(c *gin.Context) {
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("Transfer-Encoding", "chunked")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
}

func relayErrorHandler(resp *http.Response) (openAIErrorWithStatusCode *OpenAIErrorWithStatusCode) {
	openAIErrorWithStatusCode = &OpenAIErrorWithStatusCode{
		StatusCode: resp.StatusCode,
		OpenAIError: OpenAIError{
			Message: fmt.Sprintf("bad response status code %d", resp.StatusCode),
			Type:    "upstream_error",
			Code:    "bad_response_status_code",
			Param:   strconv.Itoa(resp.StatusCode),
		},
	}
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return
	}
	err = resp.Body.Close()
	if err != nil {
		return
	}
	var textResponse TextResponse
	err = json.Unmarshal(responseBody, &textResponse)
	if err != nil {
		return
	}
	openAIErrorWithStatusCode.OpenAIError = textResponse.Error
	return
}

func getFullRequestURL(baseURL string, requestURL string, channelType int) string {
	fullRequestURL := fmt.Sprintf("%s%s", baseURL, requestURL)
	if channelType == common.ChannelTypeOpenAI {
		if strings.HasPrefix(baseURL, "https://gateway.ai.cloudflare.com") {
			fullRequestURL = fmt.Sprintf("%s%s", baseURL, strings.TrimPrefix(requestURL, "/v1"))
		}
	}
	return fullRequestURL
}

func postConsumeQuota(ctx context.Context, tokenId int, quota int, userId int, channelId int, modelRatio float64, groupRatio float64, modelName string, tokenName string) {
	err := model.PostConsumeTokenQuota(tokenId, quota)
	if err != nil {
		common.SysError("error consuming token remain quota: " + err.Error())
	}
	err = model.CacheUpdateUserQuota(userId)
	if err != nil {
		common.SysError("error update user quota cache: " + err.Error())
	}
	if quota != 0 {
		logContent := fmt.Sprintf("模型倍率 %.2f，分组倍率 %.2f", modelRatio, groupRatio)
		model.RecordConsumeLog(ctx, userId, channelId, 0, 0, modelName, tokenName, quota, logContent)
		model.UpdateUserUsedQuotaAndRequestCount(userId, quota)
		model.UpdateChannelUsedQuota(channelId, quota)
	}
}
