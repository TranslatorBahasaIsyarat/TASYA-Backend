const express = require("express");
const router = express.Router();
const controller = require("../controller/index");

// Define routes
router.get("/", controller.default);

module.exports = router;
