# Render Deployment Guide

This project is split into two deployables:

1. **FastAPI backend** (`backend/`) – serves the `/query` endpoint
2. **React frontend** (`frontend/`) – static site built with Vite

The repository includes a `render.yaml` so you can deploy both services with a single blueprint import. The sections below explain the configuration in more detail and list the environment variables you must provide.

## 1. Prerequisites

- Render account with Starter plan (or higher) for both a Web Service and Static Site.
- Pinecone index + credentials (`PINECONE_API_KEY`, `PINECONE_REGION`, `PINECONE_INDEX`).
- NeonDB connection string (`NEON_DB_URI`).
- OpenAI API key (`OPENAI_API_KEY`).
- Shared access token for the app (`AUTH_TOKEN`).

## 2. Deploy with the Blueprint

1. Push the latest code to your default branch.
2. In the Render dashboard select **Deploy > Blueprint** and point it at your repository. Render will automatically pick up `render.yaml`.
3. On the configuration screen:
   - Set values for all backend `envVars`:
     - `AUTH_TOKEN`
     - `OPENAI_API_KEY`
     - `PINECONE_API_KEY`
     - `PINECONE_REGION`
     - `PINECONE_INDEX`
     - `NEON_DB_URI`
   - Optionally adjust `CORS_ALLOW_ORIGINS` to the final frontend URL if you change the service names.
4. Kick off the deploy. Render will build the backend (installing `backend/requirements.txt`) and then build the frontend (`npm install && npm run build`).

## 3. Service details

### Backend (`bnb-chat-backend`)

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Working directory:** `backend/`
- **Important env vars:** see list above. `MPLCONFIGDIR=/tmp/mpl` prevents Matplotlib from writing to unwritable home directories.

### Frontend (`bnb-chat-frontend`)

- **Build command:** `cd frontend && npm install && npm run build`
- **Publish directory:** `frontend/dist`
- **Env var:** `VITE_API_BASE_URL` should point to the backend’s public base URL (Render value is filled with the default `https://bnb-chat-backend.onrender.com`). Update it if you rename the backend service.

## 4. Manual deploy (optional)

If you prefer to configure services manually:

1. **Backend Web Service**
   - Create a new Web Service from Git.
   - Set the root directory to `backend/`.
   - Use the build/start commands above.
   - Add the environment variables under the “Environment” tab.

2. **Frontend Static Site**
   - Create a Static Site from the same repo.
   - Build command: `cd frontend && npm install && npm run build`
   - Publish directory: `frontend/dist`
   - Add `VITE_API_BASE_URL=https://<your-backend-service>.onrender.com`

## 5. Local `.env` files

For local development you can set `VITE_API_BASE_URL=/api` (see `frontend/.env.example`) and run the backend on port 8000 with the provided proxy in `vite.config.js`.

## 6. After deployment

- Update `CORS_ALLOW_ORIGINS` if the frontend ends up with a custom domain.
- Rotate the shared token by changing `AUTH_TOKEN` in Render; the frontend will force a login the next time a user visits.
- Check Render build logs the first time to confirm Pinecone and Neon connections succeed.

That’s it—import the blueprint, supply secrets, and Render handles the rest.
