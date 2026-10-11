/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        hotpink: {
          50: '#fff0f5',
          100: '#ffe4e9',
          200: '#fcc2d0',
          300: '#f783ac',
          400: '#f06595',
          500: '#e64980', // Hot Pink
          600: '#d6336c',
          700: '#c2255c',
          800: '#9b1c47',
          900: '#731333',
        }
      }
    },
  },
  plugins: [],
}
