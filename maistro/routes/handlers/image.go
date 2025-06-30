package handlers

import (
	"fmt"
	"maistro/config"
	"maistro/proxy"

	"github.com/gofiber/fiber/v2"
)

// DownloadImage proxies image download requests to the Stable Diffusion API
func DownloadImage(c *fiber.Ctx) error {
	filename := c.Params("filename")
	conf := config.GetConfig(nil)
	targetURL := fmt.Sprintf("%s/download/%s", conf.InferenceServices.StableDiffusion.BaseURL, filename)
	res, err := proxy.ProxyRequest(c.UserContext(), fiber.MethodGet, targetURL, false, c.Body())
	if err != nil {
		return handleApiError(err, fiber.StatusInternalServerError, "Failed to download image")
	}
	c.Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Set("Content-Type", "application/octet-stream")
	return c.Status(res.StatusCode).SendStream(res.Body)
}
