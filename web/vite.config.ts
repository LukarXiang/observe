import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时把 /api 代理到 observe serve（127.0.0.1:8765）；构建产物放 dist，由 FastAPI 托管
export default defineConfig({ plugins: [react()], server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8765' } } })
