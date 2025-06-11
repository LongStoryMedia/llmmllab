# Stage 1: Build the Go binary
FROM golang:1.24 AS builder

WORKDIR /app
COPY . .
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o proxyllama .

# Stage 2: Minimal runtime image
FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/proxyllama .
COPY --from=builder /app/.config.yaml .

# Expose the port the server listens on
EXPOSE 8080

CMD ["./proxyllama"]
