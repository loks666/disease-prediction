$(function () {
	"use strict";
	// chart 1
	var options = {
		series: [{
			name: 'Income',
			data: [18, 51, 80, 38, 88, 50, 40, 52, 88, 80, 60, 70]
		}, {
			name: 'Expenses',
			data: [27, 38, 60, 77, 40, 50, 49, 29, 42, 27, 42, 50]
		}],
		chart: {
			foreColor: 'rgba(255, 255, 255, 0.65)',
			type: 'area',
			height: 340,
			toolbar: {
				show: false
			},
			zoom: {
				enabled: false
			},
			dropShadow: {
				enabled: false,
				top: 3,
				left: 14,
				blur: 4,
				opacity: 0.10,
			}
		},
		legend: {
			position: 'top',
			horizontalAlign: 'left',
			offsetX: -25
		},
		dataLabels: {
			enabled: false
		},
		stroke: {
			show: true,
			width: 3,
			curve: 'smooth'
		},
		tooltip: {
			theme: 'dark',
			y: {
				formatter: function (val) {
					return "$ " + val + " "
				}
			}
		},
		fill: {
			type: 'gradient',
			gradient: {
				shade: 'light',
				gradientToColors: ['#fff', 'rgba(255, 255, 255, 0.65)'],
				shadeIntensity: 1,
				type: 'vertical',
				inverseColors: false,
				opacityFrom: 0.4,
				opacityTo: 0.1,
				//stops: [0, 50, 65, 91]
			},
		},
		grid: {
			show: true,
			borderColor: 'rgba(255, 255, 255, 0.12)',
			strokeDashArray: 5,
		},
		colors: ["#fff", "rgba(255, 255, 255, 0.65)"],
		yaxis: {
			labels: {
				formatter: function (value) {
					return value + "$";
				}
			},
		},
		xaxis: {
			categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
		}
	};
	var chart = new ApexCharts(document.querySelector("#chart1"), options);
	chart.render();
	// chart 2
	var options = {
		series: [44, 55, 60, 17],
		chart: {
			foreColor: 'rgba(255, 255, 255, 0.65)',
			height: 300,
			type: 'donut',
		},
		legend: {
			position: 'bottom',
			show: false,
		},
		plotOptions: {
			pie: {
				customScale: 0.8,
				donut: {
					size: '75%'
				}
			}
		},
		colors: ["#fff", "rgba(255, 255, 255, 0.65)", "rgba(255, 255, 255, 0.40)", "rgba(255, 255, 255, 0.25)", "rgba(255, 255, 255, 0.15)"],
		dataLabels: {
			enabled: false
		},
		tooltip: {
			enabled: false,
			theme: 'dark',
		},
		stroke:{
         colors:['rgba(255, 255, 255, 0.0)']
        },
		labels: ['Mobile', 'Desktop', 'Unknown', 'Tablet'],
		responsive: [{
			breakpoint: 480,
			options: {
				chart: {
					height: 300
				},
				legend: {
					position: 'bottom'
				},
				plotOptions: {
					pie: {
						customScale: 1,
					}
				},
			}
		}]
	};
	var chart = new ApexCharts(document.querySelector("#chart2"), options);
	chart.render();
	// chart 3
	var options1 = {
		chart: {
			foreColor: 'rgba(255, 255, 255, 0.65)',
			type: 'area',
			//width: 170,
			height: 200,
			sparkline: {
				enabled: false
			},
			toolbar: {
				show: false
			},
			zoom: {
				enabled: false
			},
			dropShadow: {
				enabled: false,
				top: 3,
				left: 14,
				blur: 4,
				opacity: 0.10,
			}
		},
		dataLabels: {
			enabled: false
		},
		fill: {
			type: 'gradient',
			gradient: {
				shade: 'light',
				gradientToColors: ['#fff'],
				shadeIntensity: 1,
				type: 'vertical',
				opacityFrom: 0.7,
				opacityTo: 0.1,
				//stops: [0, 100, 100, 100]
			},
		},
		colors: ["#fff"],
		series: [{
			name: 'Visitors',
			data: [25, 66, 41, 46, 15, 30, 59]
		}],
		stroke: {
			width: 2.5,
			curve: 'smooth',
			dashArray: [0]
		},
		grid: {
			show: true,
			borderColor: 'rgba(255, 255, 255, 0.12)',
			strokeDashArray: 5,
		},
		yaxis: {
			show: false,
		},
		tooltip: {
			theme: 'dark',
			fixed: {
				enabled: false
			},
			x: {
				show: false
			},
			y: {
				title: {
					formatter: function (seriesName) {
						return ''
					}
				}
			},
			marker: {
				show: false
			}
		}
	}
	new ApexCharts(document.querySelector("#chart3"), options1).render();
	// chart 4
	var options = {
		series: [{
			name: 'Revenue',
			data: [10, 22, 9, 12, 7, 9, 15, 19]
		}],
		chart: {
			type: 'area',
			width: 140,
			height: 65,
			sparkline: {
				enabled: true
			},
			stacked: true,
			toolbar: {
				show: false
			},
		},
		plotOptions: {
			bar: {
				horizontal: false,
				columnWidth: '25%',
				endingShape: 'rounded'
			},
		},
		legend: {
			position: 'top',
			horizontalAlign: 'left',
			offsetX: 0
		},
		dataLabels: {
			enabled: false
		},
		stroke: {
			show: true,
			width: 2,
			//colors: ['transparent']
		},
		fill: {
			type: 'gradient',
			gradient: {
				shade: 'dark',
				shadeIntensity: 0.15,
				gradientToColors: ['#fff'],
				type: 'vertical',
				inverseColors: false,
				opacityFrom: 0.1,
				opacityTo: 0.5,
				stops: [0, 50, 65, 91]
			},
		},
		colors: ["#fff"],
		xaxis: {
			categories: ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
		},
		tooltip: {
			theme: 'dark',
			y: {
				formatter: function (val) {
					return "$ " + val + " "
				}
			},
			x: {
				show: false
			},
		},
	};
	var chart = new ApexCharts(document.querySelector("#chart4"), options);
	chart.render();
	// chart 5
	var options = {
		series: [{
			name: 'Net Profit',
			data: [10, 9, 24, 19, 22, 9, 12, 7]
		}],
		chart: {
			type: 'area',
			width: 140,
			height: 65,
			sparkline: {
				enabled: true
			},
			stacked: true,
			toolbar: {
				show: false
			},
		},
		plotOptions: {
			bar: {
				horizontal: false,
				columnWidth: '25%',
				endingShape: 'rounded'
			},
		},
		legend: {
			position: 'top',
			horizontalAlign: 'left',
			offsetX: 0
		},
		dataLabels: {
			enabled: false
		},
		stroke: {
			show: true,
			width: 2,
			//colors: ['transparent']
		},
		fill: {
			type: 'gradient',
			gradient: {
				shade: 'dark',
				shadeIntensity: 0.15,
				gradientToColors: ['#fff'],
				type: 'vertical',
				inverseColors: false,
				opacityFrom: 0.1,
				opacityTo: 0.5,
				stops: [0, 50, 65, 91]
			},
		},
		colors: ["#fff"],
		xaxis: {
			categories: ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
		},
		tooltip: {
			theme: 'dark',
			y: {
				formatter: function (val) {
					return "$ " + val + " "
				}
			},
			x: {
				show: false
			},
		},
	};
	var chart = new ApexCharts(document.querySelector("#chart5"), options);
	chart.render();
	// chart 6
	var options = {
		series: [{
			name: 'Orders',
			data: [5, 9, 6, 10, 25, 9, 15, 8]
		}],
		chart: {
			type: 'area',
			width: 140,
			height: 65,
			sparkline: {
				enabled: true
			},
			stacked: true,
			toolbar: {
				show: false
			},
		},
		plotOptions: {
			bar: {
				horizontal: false,
				columnWidth: '25%',
				endingShape: 'rounded'
			},
		},
		legend: {
			position: 'top',
			horizontalAlign: 'left',
			offsetX: 0
		},
		dataLabels: {
			enabled: false
		},
		stroke: {
			show: true,
			width: 2,
			//colors: ['transparent']
		},
		fill: {
			type: 'gradient',
			gradient: {
				shade: 'dark',
				shadeIntensity: 0.15,
				gradientToColors: ['#fff'],
				type: 'vertical',
				inverseColors: false,
				opacityFrom: 0.1,
				opacityTo: 0.5,
				stops: [0, 50, 65, 91]
			},
		},
		colors: ["#fff"],
		xaxis: {
			categories: ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
		},
		tooltip: {
			theme: 'dark',
			y: {
				formatter: function (val) {
					return "$ " + val + " "
				}
			},
			x: {
				show: false
			},
		},
	};
	var chart = new ApexCharts(document.querySelector("#chart6"), options);
	chart.render();
	// chart 7
	var options = {
		series: [{
			name: 'Visitors',
			data: [4, 10, 18, 8, 17, 9, 12, 7]
		}],
		chart: {
			type: 'area',
			width: 140,
			height: 65,
			sparkline: {
				enabled: true
			},
			stacked: true,
			toolbar: {
				show: false
			},
		},
		plotOptions: {
			bar: {
				horizontal: false,
				columnWidth: '25%',
				endingShape: 'rounded'
			},
		},
		legend: {
			position: 'top',
			horizontalAlign: 'left',
			offsetX: 0
		},
		dataLabels: {
			enabled: false
		},
		stroke: {
			show: true,
			width: 2,
			//colors: ['transparent']
		},
		fill: {
			type: 'gradient',
			gradient: {
				shade: 'dark',
				shadeIntensity: 0.15,
				gradientToColors: ['#fff'],
				type: 'vertical',
				inverseColors: false,
				opacityFrom: 0.1,
				opacityTo: 0.5,
				stops: [0, 50, 65, 91]
			},
		},
		colors: ["#fff"],
		xaxis: {
			categories: ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
		},
		tooltip: {
			theme: 'dark',
			y: {
				formatter: function (val) {
					return "$ " + val + " "
				}
			},
			x: {
				show: false
			},
		},
	};
	var chart = new ApexCharts(document.querySelector("#chart7"), options);
	chart.render();
	// chart 8
	var options = {
		series: [{
			name: 'Bounce Rate',
			data: [440, 505, 414, 671, 427, 613, 901, 257, 160]
		}],
		chart: {
			type: 'area',
			height: 110,
			toolbar: {
				show: false
			},
			zoom: {
				enabled: false
			},
			dropShadow: {
				enabled: false,
				top: 3,
				left: 14,
				blur: 4,
				opacity: 0.10,
			},
			sparkline: {
				enabled: true
			}
		},
		markers: {
			size: 0,
			// colors: ["#fff"],
			strokeColors: "#fff",
			strokeWidth: 2,
			hover: {
				size: 7,
			}
		},
		dataLabels: {
			enabled: false
		},
		stroke: {
			show: true,
			width: 3,
			curve: 'smooth'
		},
		xaxis: {
			categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
		},
		fill: {
			type: 'gradient',
			gradient: {
				shade: 'light',
				gradientToColors: ['#377dff'],
				shadeIntensity: 1,
				type: 'vertical',
				inverseColors: false,
				opacityFrom: 0.4,
				opacityTo: 0.1,
				//stops: [0, 50, 65, 91]
			},
		},
		colors: ["#fff"],
		tooltip: {
			theme: 'dark',
			fixed: {
				enabled: false
			},
			x: {
				show: false
			},
			y: {
				title: {
					formatter: function (seriesName) {
						return ''
					}
				}
			},
			marker: {
				show: false
			}
		}
	};
	var chart = new ApexCharts(document.querySelector("#chart8"), options);
	chart.render();
});
jQuery('#location-map').vectorMap({
	map: 'world_mill_en',
	backgroundColor: 'transparent',
	borderColor: '#818181',
	borderOpacity: 0.25,
	borderWidth: 1,
	zoomOnScroll: false,
	color: '#009efb',
	regionStyle: {
		initial: {
			fill: '#fff'
		}
	},
	markerStyle: {
		initial: {
			r: 9,
			'fill': '#fff',
			'fill-opacity': 1,
			'stroke': '#000',
			'stroke-width': 5,
			'stroke-opacity': 0.4
		},
	},
	enableZoom: true,
	hoverColor: '#009efb',
	markers: [{
		latLng: [21.00, 78.00],
		name: 'I Love My India'
	}],
	hoverOpacity: null,
	normalizeFunction: 'linear',
	scaleColors: ['#b6d6ff', '#005ace'],
	selectedColor: '#c9dfaf',
	selectedRegions: [],
	showTooltip: true,
	onRegionClick: function (element, code, region) {
		var message = 'You clicked "' + region + '" which has the code: ' + code.toUpperCase();
		alert(message);
	}
});