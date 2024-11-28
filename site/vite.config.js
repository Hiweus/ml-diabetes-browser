import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // eslint-disable-next-line no-undef
  base: process.env.PUBLIC_URL,
  plugins: [react()],
})
