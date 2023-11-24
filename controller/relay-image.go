package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tidwall/sjson"
	"io"
	"net/http"
	"one-api/common"
	"one-api/model"
	"strings"

	"github.com/gin-gonic/gin"
)

func isWithinRange(element string, value int) bool {
	if _, ok := common.DalleGenerationImageAmounts[element]; !ok {
		return false
	}

	_min := common.DalleGenerationImageAmounts[element][0]
	_max := common.DalleGenerationImageAmounts[element][1]

	return value >= _min && value <= _max
}

func relayImageHelper(c *gin.Context, _ int) *OpenAIErrorWithStatusCode {
	imageModel := "dall-e-2"
	imageSize := "1024x1024"

	tokenId := c.GetInt("token_id")
	channelType := c.GetInt("channel")
	channelId := c.GetInt("channel_id")
	userId := c.GetInt("id")
	consumeQuota := c.GetBool("consume_quota")
	group := c.GetString("group")

	rawBody, err := common.GetBodyReusable(c)
	if err != nil {
		return errorWrapper(err, "read_request_body_failed", http.StatusInternalServerError)
	}

	var imageRequest ImageRequest
	if err := json.Unmarshal(rawBody, &imageRequest); err != nil {
		return errorWrapper(err, "bind_request_body_failed", http.StatusBadRequest)
	}

	// Size validation
	if imageRequest.Size != "" {
		imageSize = imageRequest.Size
	}

	// Model validation
	if imageRequest.Model != "" {
		imageModel = imageRequest.Model
	}

	imageCostRatio, hasValidSize := common.DalleSizeRatios[imageModel][imageSize]

	// Check if model is supported
	if hasValidSize {
		if imageRequest.Quality == "hd" && imageModel == "dall-e-3" {
			if imageSize == "1024x1024" {
				imageCostRatio *= 2
			} else {
				imageCostRatio *= 1.5
			}
		}
	} else {
		return errorWrapper(errors.New("size not supported for this image model"), "size_not_supported", http.StatusBadRequest)
	}

	// Prompt validation
	if imageRequest.Prompt == "" {
		return errorWrapper(errors.New("prompt is required"), "prompt_missing", http.StatusBadRequest)
	}

	// Check prompt length
	if len(imageRequest.Prompt) > common.DalleImagePromptLengthLimitations[imageModel] {
		return errorWrapper(errors.New("prompt is too long"), "prompt_too_long", http.StatusBadRequest)
	}

	if imageRequest.N == 0 {
		imageRequest.N = 1
	}

	// Number of generated images validation
	if isWithinRange(imageModel, imageRequest.N) == false {
		return errorWrapper(errors.New("invalid value of n"), "n_not_within_range", http.StatusBadRequest)
	}

	// map model name
	modelMapping := c.GetString("model_mapping")
	isModelMapped := false
	if modelMapping != "" {
		modelMap := make(map[string]string)
		err := json.Unmarshal([]byte(modelMapping), &modelMap)
		if err != nil {
			return errorWrapper(err, "unmarshal_model_mapping_failed", http.StatusInternalServerError)
		}
		if modelMap[imageModel] != "" {
			imageModel = modelMap[imageModel]
			isModelMapped = true
		}
	}
	baseURL := common.ChannelBaseURLs[channelType]
	requestURL := c.Request.URL.String()
	if c.GetString("base_url") != "" {
		baseURL = c.GetString("base_url")
	}

	var fullRequestURL string
	switch channelType {
	case common.ChannelTypeAzure:
		task := strings.TrimPrefix(requestURL, "/v1/")
		query := c.Request.URL.Query()
		apiVersion := query.Get("api-version")
		if apiVersion == "" {
			apiVersion = c.GetString("api_version")
		}
		fullRequestURL = fmt.Sprintf("%s/openai/deployments/%s/%s?api-version=%s", baseURL, imageRequest.Model, task, apiVersion)
	default:
		fullRequestURL = getFullRequestURL(baseURL, requestURL, channelType)
	}

	var requestBody io.Reader = c.Request.Body
	if isModelMapped {
		buf, err := sjson.SetBytes(rawBody, "model", imageRequest.Model)
		if err != nil {
			return errorWrapper(err, "set_request_body_failed", http.StatusInternalServerError)
		}
		requestBody = bytes.NewBuffer(buf)
	}

	modelRatio := common.GetModelRatio(imageModel)
	groupRatio := common.GetGroupRatio(group)
	ratio := modelRatio * groupRatio
	userQuota, err := model.CacheGetUserQuota(userId)

	quota := int(ratio*imageCostRatio*1000) * imageRequest.N

	if consumeQuota && userQuota-quota < 0 {
		return errorWrapper(errors.New("user quota is not enough"), "insufficient_user_quota", http.StatusForbidden)
	}

	req, err := http.NewRequest(c.Request.Method, fullRequestURL, requestBody)
	if err != nil {
		return errorWrapper(err, "new_request_failed", http.StatusInternalServerError)
	}
	switch channelType {
	case common.ChannelTypeAzure:
		apiKey := c.Request.Header.Get("Authorization")
		apiKey = strings.TrimPrefix(apiKey, "Bearer ")
		req.Header.Set("api-key", apiKey)
	default:
		req.Header.Set("Authorization", c.Request.Header.Get("Authorization"))
	}

	req.Header.Set("Content-Type", c.Request.Header.Get("Content-Type"))
	req.Header.Set("Accept", c.Request.Header.Get("Accept"))

	resp, err := httpClient.Do(req)
	if err != nil {
		return errorWrapper(err, "do_request_failed", http.StatusInternalServerError)
	}

	err = req.Body.Close()
	if err != nil {
		return errorWrapper(err, "close_request_body_failed", http.StatusInternalServerError)
	}
	err = c.Request.Body.Close()
	if err != nil {
		return errorWrapper(err, "close_request_body_failed", http.StatusInternalServerError)
	}
	var textResponse ImageResponse

	defer func(ctx context.Context) {
		if consumeQuota {
			err := model.PostConsumeTokenQuota(tokenId, quota)
			if err != nil {
				common.SysError("error consuming token remain quota: " + err.Error())
			}
			err = model.CacheUpdateUserQuota(userId)
			if err != nil {
				common.SysError("error update user quota cache: " + err.Error())
			}
			if quota != 0 {
				tokenName := c.GetString("token_name")
				logContent := fmt.Sprintf("模型倍率 %.2f，分组倍率 %.2f", modelRatio, groupRatio)
				model.RecordConsumeLog(ctx, userId, channelId, 0, 0, imageModel, tokenName, quota, logContent)
				model.UpdateUserUsedQuotaAndRequestCount(userId, quota)
				channelId := c.GetInt("channel_id")
				model.UpdateChannelUsedQuota(channelId, quota)
			}
		}
	}(c.Request.Context())

	if consumeQuota {
		responseBody, err := io.ReadAll(resp.Body)

		if err != nil {
			return errorWrapper(err, "read_response_body_failed", http.StatusInternalServerError)
		}
		err = resp.Body.Close()
		if err != nil {
			return errorWrapper(err, "close_response_body_failed", http.StatusInternalServerError)
		}
		err = json.Unmarshal(responseBody, &textResponse)
		if err != nil {
			return errorWrapper(err, "unmarshal_response_body_failed", http.StatusInternalServerError)
		}

		resp.Body = io.NopCloser(bytes.NewBuffer(responseBody))
	}

	for k, v := range resp.Header {
		c.Writer.Header().Set(k, v[0])
	}
	c.Writer.WriteHeader(resp.StatusCode)

	_, err = io.Copy(c.Writer, resp.Body)
	if err != nil {
		return errorWrapper(err, "copy_response_body_failed", http.StatusInternalServerError)
	}
	err = resp.Body.Close()
	if err != nil {
		return errorWrapper(err, "close_response_body_failed", http.StatusInternalServerError)
	}
	return nil
}
