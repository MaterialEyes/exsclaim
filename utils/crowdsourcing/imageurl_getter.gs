function myFunction() {
  var names=[];
  var ss=SpreadsheetApp.getActiveSpreadsheet();
  var s=ss.getActiveSheet();
  var c=s.getActiveCell();

    var fldr=DriveApp.getFolderById("FOLDER_ID");
    var files=fldr.getFiles();
    
    var f,str;
    while (files.hasNext()) {
      f=files.next();
      str="https://drive.google.com/uc?id=" + f.getId() + " " + f.getName();
      
      names.push([str]);
    }
  s.getRange(c.getRow(),c.getColumn(),names.length).setValues(names);
}