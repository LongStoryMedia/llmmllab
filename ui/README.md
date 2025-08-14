# LLM ML Lab - UI

A modern React-based user interface for interacting with the LLM ML Lab inference API services.

## Features

- Modern React implementation using TypeScript and Vite
- Material UI Joy interface components
- Real-time streaming responses for chat and completions
- Document upload and processing
- Multi-modal capabilities (text and image generation)
- Responsive design for desktop and mobile

## Tech Stack

- React with TypeScript
- Vite build system
- Material UI Joy
- ESLint with TypeScript configuration

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install
```

### Development

```bash
# Run development server
npm run dev
```

### Building for Production

```bash
# Build for production
npm run build
```

### Deployment

The UI can be deployed using the included deployment script:

```bash
./deploy.sh
```

This will build the project and deploy it according to your environment configuration.

## Environment Configuration

The project includes multiple environment configuration files:

- `.env.dev` - Development environment settings
- `.env.prod` - Production environment settings
- `.env.local` - Local overrides (not committed to version control)

## Integration with Backend Services

This UI connects to the LLM ML Lab inference services, with API endpoints configured in the environment files. See the [inference README](../inference/README.md) for backend service details.

## License

[MIT License](../LICENSE)
```
