function experiment(form, data, counterfactual_version) {


  const num_training_phases = data['training_phase'].length;

  form.addPageBreakItem()
      .setTitle('Experiment')
      .setHelpText("In this experiment, you will see videos of a single robot in different situations. Your job is to predict the robot's behavior in new situations."
        +"On the next few pages you will alternate between TRAINING and TESTING phases.  During a TRAINING phase you will be shown some videos of a robot."
        +"During a TESTING phase you will be asked questions about how the robot will act in a situation you haven't seen before."
        +"We will alternate through these phases " + num_training_phases.toString() + " times to see how many TRAINING videos are needed for people to understand a policy.");

  form.addTimeItem().setTitle('Please record the stop time. There is no need to solve this task as quickly as possible, but we would like a general sense for how long this task takes users.');

  // Assume data['training_phase'] and data['testing_phase'] are both lists of the same length.
  // Assume data['training_phase'][i] and data['testing_phase'][i] are both lists with arbitrary lengths.
  // Each element in them is one training or testing videos
  for (var i = 0; i < num_videos; i++) {

    form.addPageBreakItem()
    .setTitle('TRAINING Phase' + i.toString() + " of " + num_training_phases.toString())
    .setHelpText("Please watch the video below and try to understand how the robot "
                 +"typically acts. You can re-watch the video as many times as you would like, "
                 +"but once you proceed to the next page you will not be able to return.");

    var videos = data["training_phase"][i];
    for (var video in videos) {
      var video_explanation = form.addImageItem().setImage(videos[video]);
    }


    // Next page:
    form.addPageBreakItem()
    .setTitle('TESTING Phase' + i.toString() + " of " + num_training_phases.toString())
    .setHelpText('Please do not go back to the previous page while answering these questions.');


    if (data["use_comparison_task"]) {
      var comparison_task_list = data["testing_phase"][i];
      for (var video in videos) {
        var videos = comparison_task_list["img_options"];
        comparison_task(form, videos);
      }
    }
    if (data["use_final_position_task"]) {
      var comparison_task_list = data["testing_phase"][i];
      for (var video in videos) {
        var pre_video = comparison_task_list["pre_video"];
        var images = comparison_task_list["img_options"];
        final_position_task(form, pre_video, images);
      }
    }

  }
  form.addPageBreakItem()
      .setTitle('Additional Analysis')
      .setHelpText('Please do not go back to the previous page while answering these questions.');
  form.addTimeItem().setTitle('When you are done with this page, please record the stop time.');

//  var textValidation = FormApp.createTextValidation()
//  .setHelpText("Input was not a positive number.")
//  .requireNumberGreaterThan(0)
//  .build();
//  form.addTextItem()
//      .setTitle("How many minutes do you think you have spent on the survey so far?")
//      .setValidation(textValidation).setRequired(true);
  
  
}
