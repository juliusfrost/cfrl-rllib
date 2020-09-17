function experiment(form, data, counterfactual_version) {
  form.addPageBreakItem()
      .setTitle('Behavior Understanding')
      .setHelpText("Here is a new robot behavior. "
                      +"Please watch the videos below and try to understand how the robot "
                      +"typically acts and what problems there are (if any) with the behavior. "
                      +"All videos are from the same robot and may have the same problem. "
                      +'You can re-watch each video as many times as you would like, '
                      +"but once you proceed to the next page you will not be able to return.");

  // Record start time (so we can see how long people spend on these videos)
//  form.addTimeItem().setRequired(true)
//      .setTitle('Please record the current time so we can calculate how long you spent on this survey.')
  
  
  // Show videos
  var videos = data["training_phase"];
  for (var video in videos) {
    var video_explanation = form.addImageItem().setImage(videos[video]);
    var video_understanding = form.addScaleItem().setRequired(true);
    video_understanding.setTitle("Did you understand the robot's behavior in the video above?").setBounds(1, 10).setLabels("didn't understand", "did understand");
    var video_usefulness = form.addScaleItem().setRequired(true);
    video_usefulness.setTitle("How useful was the video above in helping you understand the overall robot's behavior?").setBounds(1, 10).setLabels("useless", "useful");
  }
  
  // Next page: 
  form.addPageBreakItem()
      .setTitle('Analysis')
      .setHelpText('Please do not go back to the previous page while answering these questions.');
    
  
  // Open-ended behavior analysis
  form.addParagraphTextItem().setRequired(true)
      .setTitle("Do you think the behavior you just saw was suboptimal? "
               +"What problem, if any, do you think the robot has? "
               +"If you think you know the reason behind the robot's suboptimality, please include it. "
               +"(e.g. in the example on the first page, please write 'the robot can only move two steps at a time' "
               +"rather than just writing 'the robot cannot reach its goal').")
    
    // TODO: consider looping this so we can add multiple
    
  var comparison_task_list = data["testing_phase"];
  for (var i = 0; i < comparison_task_list["video1"].length; i++) {
    Logger.log("TESTING PHASE");
    Logger.log(comparison_task_list);
    Logger.log(i);
    var video1 = comparison_task_list["video1"][i];
    var video2 = comparison_task_list["video2"][i];
    comparison_task(form, video1, video2);
  }
  
  form.addPageBreakItem()
      .setTitle('Additional Analysis')
      .setHelpText('Please do not go back to the previous page while answering these questions.');
  
  
  // Multiple-choice question about the behavior problems  
  var item = form.addMultipleChoiceItem().setRequired(true);
  item.setTitle("Which of these do you think is the most likely problem with the robot's behavior?");
  item.setChoices([
    item.createChoice('There is no problem with this robot.'),
    item.createChoice('The robot is programmed to visit every object, not just the goal object.'),
    item.createChoice("The robot can't turn right. When it should turn right, it keeps going straight."),
    item.createChoice('The robot periodically glitches and takes a random action.'),
    item.createChoice('The robot avoids all yellow objects.'),
    item.createChoice('The robot must explore the entire grid before going to the goal.'),
    item.createChoice("The robot can't turn right. It always turns left instead."),
    item.createChoice('The robot is blind and is moving randomly.'),
    item.createChoice('The robot explores randomly while it cannot see the goal item.'),
    
  ]); // TODO: randomize order
  
//  form.addTimeItem().setTitle('When you are done with this page, please record the stop time.');
    
  var textValidation = FormApp.createTextValidation()
  .setHelpText("Input was not a positive number.")
  .requireNumberGreaterThan(0)
  .build();
  form.addTextItem()
      .setTitle("How many minutes do you think you have spent on the survey so far?")
      .setValidation(textValidation).setRequired(true);
  
  
}
