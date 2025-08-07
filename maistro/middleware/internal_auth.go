package middleware

import (
	"maistro/config"
	"maistro/util"
	"net"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/sirupsen/logrus"
)

// InternalServiceAuth is middleware to verify requests from internal services
func InternalServiceAuth(c *fiber.Ctx) error {
	conf := config.GetConfig(nil)

	// Get the API key from the request header
	apiKey := c.Get("X-API-Key")

	// Verify the API key
	if apiKey == "" || apiKey != conf.Internal.APIKey {
		util.LogWarning("Unauthorized internal API request", logrus.Fields{
			"ip":     c.IP(),
			"path":   c.Path(),
			"method": c.Method(),
		})
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"success": false,
			"error":   "Unauthorized",
		})
	}

	// If we're restricting by IP, check that too
	// Note: You'll need to add "net" to the imports
	if conf.Internal.AllowedIPs != "" {
		clientIP := c.IP()
		allowedIPs := strings.Split(conf.Internal.AllowedIPs, ",")
		allowed := false

		for _, ipEntry := range allowedIPs {
			ipEntry = strings.TrimSpace(ipEntry)

			// Check if this is a CIDR range
			if strings.Contains(ipEntry, "/") {
				// Parse CIDR
				_, ipNet, err := net.ParseCIDR(ipEntry)
				if err != nil {
					util.LogWarning("Invalid CIDR notation in allowed IPs", logrus.Fields{
						"cidr":  ipEntry,
						"error": err.Error(),
					})
					continue
				}

				// Parse client IP
				clientIPAddr := net.ParseIP(clientIP)
				if clientIPAddr == nil {
					util.LogWarning("Invalid client IP", logrus.Fields{
						"ip": clientIP,
					})
					continue
				}

				// Check if client IP is in CIDR range
				if ipNet.Contains(clientIPAddr) {
					allowed = true
					break
				}
			} else {
				// Direct IP comparison
				if ipEntry == clientIP {
					allowed = true
					break
				}
			}
		}

		if !allowed {
			util.LogWarning("IP not allowed for internal API", logrus.Fields{
				"ip":     clientIP,
				"path":   c.Path(),
				"method": c.Method(),
			})
			return c.Status(fiber.StatusForbidden).JSON(fiber.Map{
				"success": false,
				"error":   "IP not allowed",
			})
		}
	}

	return c.Next()
}
