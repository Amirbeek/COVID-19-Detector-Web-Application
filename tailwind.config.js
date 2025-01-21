module.exports = {
  content: [
    './templates/**/*.html',
    './static/**/*.js'
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('tailwindcss'),
    require('autoprefixer'),
  ]
};
