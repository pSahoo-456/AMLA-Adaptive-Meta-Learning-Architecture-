/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0f172a', // slate-900
        surface: '#1e293b', // slate-800
        primary: '#6366f1', // indigo-500
        secondary: '#8b5cf6', // violet-500
        accent: '#10b981', // emerald-500
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
