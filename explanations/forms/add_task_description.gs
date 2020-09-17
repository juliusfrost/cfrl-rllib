function add_task_description(form, data, counterfactual_version) {
  // Describe task
  var section_title = form.addSectionHeaderItem();
  section_title.setTitle("The Robot Task");
  section_title.setHelpText("The red triangle is a robot trying to find objects in a gridworld. "
                           +"The robot can move forward and turn left or right. "
                           +"The robot has succeeded when it is adjacent to the goal object "
                           +"and is facing it. The gray area in front of the robot "
                           +"is the region visible to the robot's sensors.");
  var img = DriveApp.getFileById('1Tx94475CdzQ02q5-c_SJb4nnMeyn27H_');
  // Show image of the gridworld.
  var image_item = form.addImageItem().setImage(img).setTitle("Example Gridworld and Task");
  
  add_video_description(form, data);
  if (counterfactual_version) {
    add_counterfactual_description(form, data);
  }
  
  form.addSectionHeaderItem()
      .setHelpText("On the next page, we will show you another (possibly) suboptimal robot behavior "
                   +"and ask you some questions about it.");
}


function add_counterfactual_description(form, data) {
  var bot_desc = form.addImageItem().setImage(DriveApp.getFileById('1CLOTZQnP1-SCcYFlMxaz8JakiPCqMWhQ'));
  bot_desc.setTitle("There are 3 types of robot behaviors. "
                    +"The red triangle is the robot following the original path it would take. "
                    +"The yellow triangle is a user trying to intervene to see what would happen if things went differently. "
                    +"The purple triangle is the robot taking over control after the user intervenes.");
  for (var id in data["instruction_phase"]["example_counterfactual"]) {
    var bot_video = data["instruction_phase"]["example_counterfactual"][id];
    var img = form.addImageItem().setImage(bot_video);
    if (id == 0) {
      // Overlay View
      img.setTitle("Here are some examples of alternative paths the robot could take (yellow/purple), "
                   +"to aid your understanding in the original behavior of the robot (red). "
                   +"In the videos shown below, the original robot starts moving along a path (red). "
                   +"At some point, the user intervenes to see what would happen if a different path was taken (yellow). "
                   +"Then the robot regains control to finish the task (purple). "
                   +"Please take your time studying the difference between the original path and alternative path until you are comfortable.");
    }
  }
}


function add_video_description(form, data) {
  // Add expert demos
  for (var id in data["instruction_phase"]["example_expert"]) {
    var bot_video = data["instruction_phase"]["example_expert"][id];
    var img = form.addImageItem().setImage(bot_video);
    if (id == 0) {
      img.setTitle("Here are a two examples of an expert robot solving the task.");
    }
  }
  
  // Add suboptimal demos
  for (var id in data["instruction_phase"]["example_suboptimal"]) {
    var bot_video = data["instruction_phase"]["example_suboptimal"][id];
    var img = form.addImageItem().setImage(bot_video);
    if (id == 0) {
      img.setTitle("For comparison, here is an example of one possible suboptimal robot behavior. "
                 +"All suboptimal robots you see will have a single, human-understandable problem. "
                 +"The problem with this robot is that it is programmed to always take 2 steps "
                 +"forward instead of 1. "
                 +"As a result, the robot sometimes cannot reach its goal location. "
                 +"Both videos below are from the same robot. "
             );
    }
  }
}
