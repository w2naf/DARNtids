function populateEndYears() {
	var syear = 1993;
	var ayear = (new Date()).getFullYear();
	var nyears = ayear - syear;
	var yearNames = new Array(nyears);
	var selected = new Array(nyears);
	for (var i=nyears; i>=0; i--) {
		yearNames[i] = String(ayear-i);
		if (ayear-i == ayear) {
			selected[i] = 1;
		} else {
			selected[i] = 0;
		}
	}
	populateSelectElement(yearNames, yearNames, selected, document.getElementById("endYr"));
}

function populateEndMonths() {
	var monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
	var monthValues = new Array(12);
	var selected = new Array(12);
	var amonth = (new Date()).getMonth();
	for (var i=0; i<=11; i++) {
		if (i+1 < 10) {
			monthValues[i] = "0"+String(i+1);
		} else {
			monthValues[i] = String(i+1);
		}
		if (i == amonth) {
			selected[i] = 1;
		} else {
			selected[i] = 0;
		}
	}
	populateSelectElement(monthNames, monthValues, selected, document.getElementById("endMon"));
}

function populateEndDays() {
	var adate = new Date();
	var ayear = adate.getFullYear();
	var amonth = adate.getMonth();
	var aday = adate.getDate();
	var ndays = daysInMonth(ayear, amonth);
	var dayNames = new Array(ndays);
	var selected = new Array(ndays);
	for (var i=0; i<ndays; i++) {
		if (i+1 < 10) {
			dayNames[i] = "0"+String(i+1);
		} else {
			dayNames[i] = String(i+1);
		}
		if (i+1 == aday) {
			selected[i] = 1;
		} else {
			selected[i] = 0;
		}
	}
	populateSelectElement(dayNames, dayNames, selected, document.getElementById("endDay"));
}

function populateLists() {
	div = document.getElementById("radarN");
	if (div != null) {
		ipPopulateNorthernRadars();
	}
	div = document.getElementById("radarS");
	if (div != null) {
		ipPopulateSouthernRadars();
	}
	div = document.getElementById("yr");
	if (div != null) {
		populateYears();
	}
	div = document.getElementById("mon");
	if (div != null) {
		populateMonths();
	}
	div = document.getElementById("day");
	if (div != null) {
		populateDays();
	}
	div = document.getElementById("endYr");
	if (div != null) {
		populateEndYears();
	}
	div = document.getElementById("endMon");
	if (div != null) {
		populateEndMonths();
	}
	div = document.getElementById("endDay");
	if (div != null) {
		populateEndDays();
	}
}

function updateEndDayList() {
	var ayear = document.getElementById("endYr").value;
	var amonth = document.getElementById("endMon").value - 1;
	var selectedDay = document.getElementById("endDay").selectedIndex;
	var ndays = daysInMonth(ayear, amonth);
	var dayNames = new Array(ndays);
	var selected = new Array(ndays);
	for (var i=0; i<ndays; i++) {
		if (i+1 < 10) {
			dayNames[i] = "0"+String(i+1);
		} else {
			dayNames[i] = String(i+1);
		}
		if (i == selectedDay) {
			selected[i] = 1;
		} else {
			selected[i] = 0;
		}
	}
	removeChildren("endDay");
	populateSelectElement(dayNames, dayNames, selected, document.getElementById("endDay"));
}


function ipConstructURL(type) {
	if (type == "asciipot") {
		return ipConstructASCIIPOTURL();
	}
}

////////////////////////////////////////////////////////////////////////////////
// ASCII Potential URL /////////////////////////////////////////////////////////

function ipConstructASCIIPOTURL() {
	shr = document.plotForm.stHr.value;
	smin = document.plotForm.stMin.value;
	
	ehr = document.plotForm.endHr.value;
	emin = document.plotForm.endMin.value;

	var hem_value = "";
	for(var i=0; i < document.plotForm.hemisphere.length; i++) {
		if(document.plotForm.hemisphere[i].selected) {
			hem_value = document.plotForm.hemisphere[i].value;
			break;
		}
	}
	if(hem_value == "") {
		alert("Please select a hemisphere.");
		return null;
	}

	deb_value = "0";
	if (document.getElementById("debug").checked)
		deb_value = "1";

	var url = "/assets/php/ascii_plot_lib.php?ASCIIPOT=yes"+
		"&date="+document.plotForm.yr.value+
			document.plotForm.mon.value+
			document.plotForm.day.value+
//		"&enddate="+document.plotForm.endYr.value+
//			document.plotForm.endMon.value+
//			document.plotForm.endDay.value+
		"&shr="+shr+
		"&smin="+smin+
		"&ehr="+ehr+
		"&emin="+emin+
		"&hemi="+hem_value+
		"&timeStep="+document.plotForm.timeStep.value+
		"&debug="+deb_value;
	return url;
}

// End ASCII Potential URL /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
