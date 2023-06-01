const express = require("express");
const app = express();

const routes = require("./routes/index");

// Set up routes
app.use("/", routes);

// Start the server
const port = 3000;
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
