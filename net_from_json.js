/*
function openCity(evt, cityName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
          tabcontent[i].style.display = "none";
        }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
          tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
    document.getElementById(cityName).style.display = "block";
    evt.currentTarget.className += " active";
}
*/
console.log("Coucou!")
if (typeof jQuery == 'undefined'){
  console.log("shit");
  }
else {
  console.log("wouhou!");
}
$.getJSON("./saved_model/test_model.json",function(json) {
  console.log(json);
  for (x in json) {
    //add the tab button
    var tab = document.createElement("button");
    tab.className = "tablinks";
    tab.innerText = json[x].name;
    //necessary to "fix" the json[x].name on a constant
    const k = json[x].name;
    tab.onclick = function(ev) {openCity(ev,k);};
    document.getElementById("tab").appendChild(tab);
    
    //add the div
    var diva = document.createElement("div");
    diva.className = "tabcontent";
    diva.id = json[x].name;
    var title = document.createElement("h3");
    var textnode = document.createTextNode(json[x].name);
    title.appendChild(textnode);
    diva.appendChild(title);

    //add the pics
    if ("filters" in json[x]){
      var cent = document.createElement("div");
      cent.className = "Center";
      var row = document.createElement("div");
      row.className = "Row";
      for (j = 0; j < 3; j++){
        var col = document.createElement("div");
        col.className = "Column";
        for (i = 0;i < 3;i++) {
          var link = document.createElement("a");
          link.href = "img_folder/test1.JPEG";
          var image = document.createElement("img");
          const p = json[x].filters[(i+j)].actmax_im;
          image.src = p;
          console.log(json[x].filters[(i+j)].fav_im);
          link.appendChild(image);
          col.appendChild(link);
        }
      row.appendChild(col);
      }
      cent.appendChild(row);
      diva.appendChild(cent);
    }
    document.getElementById("wrapper").appendChild(diva);

  }
});

function openCity(evt, cityName) {
    console.log("openCity!"+cityName);
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
          tabcontent[i].style.display = "none";
        }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
          tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
    document.getElementById(cityName).style.display = "block";
    evt.currentTarget.className += " active";
}

