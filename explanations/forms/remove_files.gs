function remove_files(directory, name) {
  var files = directory.getFiles();
  while (files.hasNext()) {
    var file = files.next();
    var file_name = file.getName();
    Logger.log(file_name);
    if (file_name == name) {
      Logger.log("removing file");
      DriveApp.removeFile(file);
    }
  }
}
