<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    .box {
      width: 200px;
      height: 100px;
      border: solid 1px #999;
      margin: 5px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .arrow {
      width: 24px;
      height: 24px;
      background-color: #999;
      border-radius: 50%;
      position: absolute;
      right: -12px;
      bottom: 18px;
    }
    .rotate-90 {
      transform: rotate(90deg);
    }
  </style>
</head>
<body>
  <div class="box">
    <div>Client</div>
    <svg class="arrow"></svg>
  </div>
  <div class="box">
    <div>Proxy Service</div>
    <svg class="arrow rotate-90"></svg>
  </div>
  <div class="box">
    <div>LLM (Language Model)</div>
    <svg class="arrow rotate-90"></svg>
  </div>
  <div class="box">
    <div>PostgreSQL</div>
  </div>
  <div class="box">
    <div>Kubernetes Nodes</div>
  </div>

  ---

  <div class="box">
    <div>Data Access Decision</div>
    <svg class="arrow"></svg>
  </div>
  <div class="box">
    <div>Analyze Query Context</div>
    <svg class="arrow rotate-90"></svg>
  </div>
  <div class="box">
    <div>Determine Need for Data</div>
  </div>
  <div class="box">
    <div>Retrieve Data (if necessary)</div>
  </div>
</body>
</html>