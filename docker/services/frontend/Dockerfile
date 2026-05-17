# Frontend doesn't strictly need a custom base since nginx:alpine is tiny
FROM nginx:alpine

COPY index.html /usr/share/nginx/html/index.html
COPY nginx.conf /etc/nginx/conf.d/default.conf
