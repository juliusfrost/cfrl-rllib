function comparison_task(form, video_list) {
  // View the videos below.  Which came from the policy you observed on the previous page?
  var item = form.addMultipleChoiceItem().setRequired(true);
  const num_videos = video_list.length;
  const num_videos_str = num_videos.toString();
  item.setTitle("Which of the " + num_videos_str + "videos below do you think came from the robot you saw on the previous page?");
  var choices = [];
  for (var i = 0; i < num_videos; i++) {
    choices.push(item.createChoice('Video ' + num_videos_str));
    form.addImageItem()
    .setImage(video_list[i])
    .setTitle("Video " + num_videos_str);
  }

  item.setChoices(choices);
}