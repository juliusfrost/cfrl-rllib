function comparison_task(form, video1, video2) {
  // View the two videos below.  Which came from the policy you observed on the previous page?
  var item = form.addMultipleChoiceItem().setRequired(true);
  item.setTitle("Which of the two videos below do you think came from the robot you saw on the previous page?");
  item.setChoices([
    item.createChoice('Video 1'),
    item.createChoice('Video 2'),
  ]);
    
  form.addImageItem()
    .setImage(video1)
    .setTitle("Video 1");
  
  form.addImageItem()
    .setImage(video2)
    .setTitle("Video 2");
  
}
