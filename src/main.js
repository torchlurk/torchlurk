console.log("Coucou!")

if (typeof jQuery == 'undefined'){
  console.log("shit");
  }
else {
  console.log("wouhou!");
}


$.getJSON("../saved_model/test_model.json",function(json) {
  console.log(json);
  const nearest_sq = n => Math.round(Math.sqrt(n));
  for (x in json) {
    const FAV_TARGET_CLASS = "fav_hover" + json[x].ind.toString();
    const GRAD_TARGET_CLASS = "grad_hover" + json[x].ind.toString();
    const EXT_TARGET = "extend_hover" +json[x].ind.toString();
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
    var title = document.createElement("h2");
    title.className = "Center";
    var textnode = document.createTextNode(json[x].name);
    title.appendChild(textnode);
    diva.appendChild(title);
    
    //extend images
    //TODO: make the loop adapt to the number of favourites images in the josn
    for (var i=0;i<7;i++){
      var bigimg = document.createElement("img");
      bigimg.src = "../data/tinyimagenet/numb.png";
      bigimg.style.width = "15%";
      bigimg.style.height = "15%";
      diva.appendChild(bigimg);
      if (i == 0){
        bigimg.id = EXT_TARGET;
      }
      else if (1 <=i && i <=3){
        bigimg.className = FAV_TARGET_CLASS;
      }
      else {
        bigimg.className = GRAD_TARGET_CLASS;
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
            const fav_srcs = json[x].filters[curr_indx].fav_imgs;
            const grad_srcs = json[x].filters[curr_indx].grad_path;
            image.addEventListener('mouseover', function() {
              extend_id_img(ext_src,EXT_TARGET);
              extend_class_img(fav_srcs,FAV_TARGET_CLASS);
              extend_class_img(grad_srcs,GRAD_TARGET_CLASS);
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

function extend_id_img(src,target) {
  console.log("SOURCE",src);
  console.log("TARGET",target);
  ext_img = document.getElementById(target);
  ext_img.src = src;  
}

function extend_class_img(srcs,target_class) {
  console.log("SOURCE",srcs);
  console.log("TARGET",target_class);
  ext_img = document.getElementsByClassName(target_class);
  console.log(ext_img);
  for (var i = 0;i<ext_img.length;i++){
    ext_img[i].src = srcs[i];
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

