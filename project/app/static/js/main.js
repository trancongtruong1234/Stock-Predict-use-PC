function handleFileUpload() {
  const csrftoken = document.getElementById("csrfmiddlewaretoken").value;
  const formData = new FormData();
  const input = document.getElementById("file");
  formData.append("file", input.files[0]);
  fetch("http://127.0.0.1:8000/handleFileUpload", {
    method: "POST",
    headers: { "X-CSRFToken": csrftoken },
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      let select = document.getElementById("comboboxPriceIndex");
      data.priceIndex.forEach((item, index) => {
        let option = document.createElement("option");
        option.value = index;
        option.text = item;
        select.appendChild(option);
      });
    })
    .catch((error) => console.log(error));
}

function showToast() {
  const train_test = document.getElementById("num3").value;
  const priceIndex = document.getElementById("comboboxPriceIndex").value;
  const csrftoken = document.getElementById("csrfmiddlewaretoken").value;
  const formData = new FormData();
  formData.append("train_test", train_test);
  formData.append("priceIndex", priceIndex);
  fetch("http://127.0.0.1:8000/train-test", {
    method: "POST",
    headers: { "X-CSRFToken": csrftoken },
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
      document.getElementById(
        "toast-body"
      ).innerHTML = `Train: ${data.train_row} rows, ${data.train_column} columns
      Test: ${data.test_row} rows, ${data.test_column} columns`;
      const toast = document.getElementById("toast");
      const newToast = new bootstrap.Toast(toast);
      newToast.show();
    })
    .catch((error) => console.log(error));
}

async function displayChart() {
  const list = document.getElementById("result").innerHTML;
  const result = JSON.parse(list);
  console.log(result)
  await google.charts.load("current", {
    packages: ["corechart", "line"],
  });
  await google.charts.setOnLoadCallback(drawLineColors);
  function drawLineColors() {
    var data = new google.visualization.DataTable();
    data.addColumn("string", "X");
    data.addColumn("number", "Giá dự đoán");
    data.addColumn("number", "Giá thực tế");

    data.addRows(result);

    var options = {
      hAxis: {
        title: "Thời gian",
      },
      vAxis: {
        title: "Giá",
      },
      colors: ["#a52714", "#097138"],
      width: 1000,
      height: 400,
    };

    var chart = new google.visualization.LineChart(
      document.getElementById("chart_div")
    );
    chart.draw(data, options);
  }
}
