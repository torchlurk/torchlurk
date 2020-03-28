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
  const nearest_sq = n => Math.round(Math.sqrt(n));
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

    //extend images
    for (var i=0;i<4;i++){
      var bigimg = document.createElement("img");
      bigimg.src = "./img_folder/test7.JPEG";
      bigimg.style.width = "15%";
      bigimg.style.height = "15%";
      diva.appendChild(bigimg);
      if (i == 0){
        bigimg.id = "extend_hover"+json[x].ind.toString();
      }
      else{
        bigimg.className = "fav_hover" + json[x].ind.toString();
      }
    }
    //add the pics
    if ("filters" in json[x]){
      var cent = document.createElement("div");
      cent.className = "Center";
      var row = document.createElement("div");
      row.className = "Row";
      
      const n_dim = nearest_sq(json[x].n_output);
      const perc = (Math.round(50/n_dim*100)/100).toString() + "%"
      
      for (j = 0; j < n_dim; j++){
        var col = document.createElement("div");
        col.className = "Column";
        for (i = 0;i < n_dim;i++) {
          var curr_indx = j * n_dim + i;
          if (curr_indx< json[x].n_output){
            var link = document.createElement("a");
            link.href = "img_folder/test1.JPEG";
            var image = document.createElement("img");
            const p = json[x].filters[curr_indx].actmax_im;
            image.src = p;

            image.style.width = perc;
            image.style.height = perc;
            const ext_src = image.src;
            const ext_target = "extend_hover" +json[x].ind.toString();
            const fav_src = json[x].filters[curr_indx].fav_im;
            const fav_target = "fav_hover" + json[x].ind.toString();
            image.addEventListener('mouseover', function() {
              extend_hover(ext_src,ext_target);
              extend_fav(fav_src,fav_target);
                });
            //console.log(json[x].filters[(i+j)].fav_im);
            link.appendChild(image);
            col.appendChild(link);
          }
        }
      row.appendChild(col);
      }
      cent.appendChild(row);
      diva.appendChild(cent);
    }
    document.getElementById("wrapper").appendChild(diva);

  }
});

function extend_hover(src,target) {
  console.log("SOURCE",src);
  console.log("TARGET",target);
  ext_img = document.getElementById(target);
  ext_img.src = src;  
}

function extend_fav(src,target) {
  console.log("SOURCE",src);
  console.log("TARGET",target);
  ext_img = document.getElementsByClassName(target);
  console.log(ext_img);
  for (var i = 0;i<ext_img.length;i++){
    ext_img[i].src = src;
  }
  for (var i = 0;i<ext_img.length;i++){
    ext_img[i].src = src;
  }
}

function openCity(evt, cityName) {
    //console.log("openCity!"+cityName);
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

