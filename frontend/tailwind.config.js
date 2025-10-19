export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          primary: "#4f46e5",
        },
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};
