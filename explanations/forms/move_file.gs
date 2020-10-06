// Credit here: https://stackoverflow.com/questions/42938990/google-sheets-api-create-or-move-spreadsheet-into-folder
function moveToCurrentFolder(form) {
  // Create a new form in drive root
  var formFile = DriveApp.getFileById(form.getId())

  // Get the parent folders of the open document
  var curr_folder = get_curr_folder().getFoldersByName("new_forms").next();

  // Check for root folder
  if (curr_folder.getId() == DriveApp.getRootFolder().getId()) return;
  
  // Add the created form to current folder
  // and remove it from root folder
  curr_folder.addFile(formFile);
  DriveApp.removeFile(formFile);
}