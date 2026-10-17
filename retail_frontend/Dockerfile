# Retail Frontend -- its own Node image, a THIRD real container alongside retail-api/
# retail-email-worker (see docker-compose.retail.yml), never the Python image those two share.
# No shared layer, process, or dependency with anything in the repo root's Dockerfile.
#
# Not proven with a real `docker build`/`npm install` on this dev machine -- Node itself isn't
# installed here (verified directly: `node --version` fails), same disclosed limitation the repo
# root's own Dockerfile already states for its own untested-in-this-environment build.
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

# Real Next.js gotcha, not glossed over: `NEXT_PUBLIC_*` vars are inlined into the client JS
# bundle at BUILD time (`next build`), not read from the environment at container start --
# setting `NEXT_PUBLIC_RETAIL_API_BASE_URL` under `environment:` in docker-compose.retail.yml
# alone would silently have no effect on what a visitor's browser actually calls. A build ARG,
# exported as ENV before `npm run build` runs, is what actually reaches the bundle.
ARG NEXT_PUBLIC_RETAIL_API_BASE_URL=http://127.0.0.1:8100
ENV NEXT_PUBLIC_RETAIL_API_BASE_URL=${NEXT_PUBLIC_RETAIL_API_BASE_URL}
RUN npm run build

FROM node:20-alpine

WORKDIR /app
ENV NODE_ENV=production

# `output: "standalone"` (next.config.js) produces a pruned server bundle -- only what's actually
# needed at runtime, not the full builder stage's node_modules/devDependencies.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
