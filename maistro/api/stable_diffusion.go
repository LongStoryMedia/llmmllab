package api

import (
	"bytes"
	"fmt"
	"io"
	"maistro/config"
	"maistro/util"
	"net/http"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// RegisterStableDiffusionRoutes adds stable diffusion related endpoints
func RegisterStableDiffusionRoutes(app *fiber.App) {
	// Image generation routes
	app.Post("/api/images/generate", ProxyImageGeneration)
	app.Get("/api/images/download/:filename", ProxyImageDownload)

	// Model management routes
	app.Get("/api/sd-models", ProxyListSDModels)
	app.Get("/api/sd-models/:modelId", ProxyGetSDModel)
	app.Post("/api/sd-models", ProxyAddSDModel)
	// app.Delete("/api/sd-models/:modelId", ProxyRemoveSDModel)
	app.Put("/api/sd-models/active/:modelId", ProxySetActiveSDModel)

	// LoRA management routes
	app.Get("/api/loras", ProxyListLoras)
	app.Get("/api/loras/:loraId", ProxyGetLora)
	app.Post("/api/loras", ProxyAddLora)
	// app.Delete("/api/loras/:loraId", ProxyRemoveLora)
	app.Put("/api/loras/:loraId/activate", ProxyActivateLora)
	app.Put("/api/loras/:loraId/deactivate", ProxyDeactivateLora)
	app.Put("/api/loras/:loraId/weight", ProxySetLoraWeight)
}

// getStableDiffusionBaseURL returns the base URL for the stable diffusion service
func getStableDiffusionBaseURL() string {
	conf := config.GetConfig(nil)
	return conf.InferenceServices.StableDiffusion.BaseURL
}

// Generic proxy function for stable diffusion API
func proxyStableDiffusionRequest(c *fiber.Ctx, path string, method string, bodyContent []byte) error {
	baseURL := getStableDiffusionBaseURL()
	targetURL := fmt.Sprintf("%s%s", baseURL, path)

	// Log the request
	util.LogInfo("Proxying request to Stable Diffusion API", logrus.Fields{
		"method": method,
		"path":   path,
		"url":    targetURL,
	})

	// Create a request to the Stable Diffusion API
	req, err := http.NewRequestWithContext(c.UserContext(), method, targetURL, bytes.NewReader(bodyContent))
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to create proxy request")
	}

	// Copy headers from original request
	c.Request().Header.VisitAll(func(key, value []byte) {
		k := string(key)
		v := string(value)
		if k != "host" && k != "connection" {
			req.Header.Set(k, v)
		}
	})

	// Set content type if there's a body
	if len(bodyContent) > 0 {
		req.Header.Set("Content-Type", "application/json")
	}

	// Create HTTP client with a longer timeout for image generation
	client := &http.Client{
		Timeout: 120 * time.Second, // Images might take longer to generate
	}

	// Make the request
	resp, err := client.Do(req)
	if err != nil {
		return handleError(err, fiber.StatusBadGateway, "Failed to contact Stable Diffusion API")
	}
	defer resp.Body.Close()

	// Read the response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return handleError(err, fiber.StatusInternalServerError, "Failed to read Stable Diffusion API response")
	}

	// If the API returns an error, pass it through
	if resp.StatusCode >= 400 {
		util.LogWarning("Stable Diffusion API error", logrus.Fields{
			"statusCode": resp.StatusCode,
			"path":       path,
		})
		return c.Status(resp.StatusCode).Send(body)
	}

	// Return the response to the client
	c.Set("Content-Type", resp.Header.Get("Content-Type"))
	return c.Status(resp.StatusCode).Send(body)
}

// ProxyImageGeneration proxies image generation requests to the Stable Diffusion API
func ProxyImageGeneration(c *fiber.Ctx) error {
	body := c.Body()
	return proxyStableDiffusionRequest(c, "/generate-image", "POST", body)
}

// ProxyImageDownload proxies image download requests to the Stable Diffusion API
func ProxyImageDownload(c *fiber.Ctx) error {
	filename := c.Params("filename")
	return proxyStableDiffusionRequest(c, "/download/"+filename, "GET", nil)
}

// ProxyListSDModels proxies model list requests to the Stable Diffusion API
func ProxyListSDModels(c *fiber.Ctx) error {
	return proxyStableDiffusionRequest(c, "/models/", "GET", nil)
}

// ProxyGetSDModel proxies get model details requests to the Stable Diffusion API
func ProxyGetSDModel(c *fiber.Ctx) error {
	modelId := c.Params("modelId")
	return proxyStableDiffusionRequest(c, "/models/"+modelId, "GET", nil)
}

// ProxyAddSDModel proxies add model requests to the Stable Diffusion API
func ProxyAddSDModel(c *fiber.Ctx) error {
	body := c.Body()
	return proxyStableDiffusionRequest(c, "/models/", "POST", body)
}

// ProxyRemoveSDModel proxies remove model requests to the Stable Diffusion API
func ProxySetActiveSDModel(c *fiber.Ctx) error {
	modelId := c.Params("modelId")
	return proxyStableDiffusionRequest(c, "/models/active/"+modelId, "PUT", nil)
}

// ProxyListLoras proxies LoRA list requests to the Stable Diffusion API
func ProxyListLoras(c *fiber.Ctx) error {
	return proxyStableDiffusionRequest(c, "/loras/", "GET", nil)
}

// ProxyGetLora proxies get LoRA details requests to the Stable Diffusion API
func ProxyGetLora(c *fiber.Ctx) error {
	loraId := c.Params("loraId")
	return proxyStableDiffusionRequest(c, "/loras/"+loraId, "GET", nil)
}

// ProxyAddLora proxies add LoRA requests to the Stable Diffusion API
func ProxyAddLora(c *fiber.Ctx) error {
	body := c.Body()
	return proxyStableDiffusionRequest(c, "/loras/", "POST", body)
}

// ProxyRemoveLora proxies remove LoRA requests to the Stable Diffusion API
func ProxyActivateLora(c *fiber.Ctx) error {
	loraId := c.Params("loraId")
	return proxyStableDiffusionRequest(c, "/loras/"+loraId+"/activate", "PUT", nil)
}

// ProxyDeactivateLora proxies deactivate LoRA requests to the Stable Diffusion API
func ProxyDeactivateLora(c *fiber.Ctx) error {
	loraId := c.Params("loraId")
	return proxyStableDiffusionRequest(c, "/loras/"+loraId+"/deactivate", "PUT", nil)
}

// ProxySetLoraWeight proxies set LoRA weight requests to the Stable Diffusion API
func ProxySetLoraWeight(c *fiber.Ctx) error {
	loraId := c.Params("loraId")
	body := c.Body()
	return proxyStableDiffusionRequest(c, "/loras/"+loraId+"/weight", "PUT", body)
}
