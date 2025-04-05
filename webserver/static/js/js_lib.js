$SCRIPT_ROOT = {{ request.script_root|tojson|safe }};

function fold(id) {
    dd = document.getElementById(id);
    if (dd.style.display == "none")     {
        dd.style.display = "block";
    } else {
        dd.style.display = "none";
    }
}

function plotRTI(radar,day,param) {
    $("#rtiImg").attr("src", "static/img/loading.gif");
    $.getJSON($SCRIPT_ROOT + '/rti', {
      radar: radar,
      gwDay: day,
      param: param
    }, function(data) {
      $("#rtiImg").attr("src", data.result);
    });
}
