FROM node:18-alpine as builder

WORKDIR /app

# Copiar package.json
COPY package.json package-lock.json ./

# Instalar dependencias
RUN npm ci

# Copiar código
COPY . .

# Build
RUN npm run build

# Servir estático con nginx (opcional, para producción)
# Para desarrollo, simplemente exponemos
EXPOSE 3000

CMD ["npm", "start"]
