const NUM_EXAMPLE_EXPERT = 2;
const NUM_EXAMPLE_SUBOPTIMAL = 2;
const NUM_EXAMPLE_COUNTERFACTUAL = 2;
const NUM_TRAINING = 7;
// const NUM_COMPARISONS = 3; // each

function generate_all_surveys() {
  /**
   * Generate all of the surveys.  We'll have one with counterfactuals and one without for each top-level data folder.
   * Expects all data to be in a folder called `videos`.
  */
  
  
  var curr_folder = get_curr_folder().getFoldersByName('videos').next();
  var expert_traj_folder = curr_folder.getFoldersByName('original-Bot').next();
  var files = expert_traj_folder.getFoldersByName('explanation-Bot').next()
                                .getFoldersByName('original').next()
                                .getFiles();
  var start_index = 7;
  var expert_trajs = iter_to_list(files).slice(start_index, start_index + NUM_EXAMPLE_EXPERT);
  
//  var example_suboptimal_name = 'DoubleForwardBot';
//  var suboptimal_traj_folder = get_curr_folder().getFoldersByName(example_suboptimal_name).next();
//  var files = suboptimal_traj_folder.getFiles();
//  var suboptimal_trajs = iter_to_list(files).slice(0, NUM_EXAMPLE_SUBOPTIMAL);
//  
  var suboptimal_trajs = [DriveApp.getFileById('1ma6yGghiNLq_T3lkoxAFxfStGVHouAXf'), DriveApp.getFileById('1mmPXRu7331j4SQBQSJxBvrj31Y6q0lX0')];
  
  var counterfactual_traj_folder = expert_traj_folder.getFoldersByName('explanation-Bot').next()
                                                     .getFoldersByName('explanation').next();
  var files = counterfactual_traj_folder.getFiles();
  var counterfactual_trajs = iter_to_list(files).slice(0, NUM_EXAMPLE_COUNTERFACTUAL);
  
  var explanation_videos = {
    "example_expert": expert_trajs,
    "example_suboptimal": suboptimal_trajs,
    "example_counterfactual": counterfactual_trajs,
  };
  
  
  var index = 0;
  // Control surveys
  var bots = curr_folder.getFolders();
  while (bots.hasNext()) {
    var bot_files = bots.next();
    if (bot_files.getName() == "original-Bot") {
      continue; // Don't generate one with the official bot.
    }
    var bot_name = bot_files.getName();
    bot_name = bot_name.slice(bot_name.indexOf("-") + 1, bot_name.length);
    setup_survey(bot_files, bot_name, false, index, explanation_videos);
    index++;
  }
  // Experimental surveys
  var bots = curr_folder.getFolders();
  while (bots.hasNext()) {
    var bot_files = bots.next();
//    if (bot_files.getName() == "original-Bot") {
//      continue; // Don't generate one with the official bot.
//    }
    var bot_name = bot_files.getName();
    bot_name = bot_name.slice(bot_name.indexOf("-") + 1, bot_name.length);
    setup_survey(bot_files, bot_name, true, index, explanation_videos);
    index++;
  }
}
 

function setup_survey(bot_files, bot_name, counterfactual_version, index, explanation_videos) {
  /** Generate a single survey. */
  var num_other_bots = iter_to_list(bot_files.getFolders()).length - 1;
  if (counterfactual_version) {
    Logger.log("counterfactual" + index.toString())
    Logger.log(bot_name)
    Logger.log(bot_files.getFoldersByName("explanation-" + bot_name).next().getName())
    Logger.log(bot_files.getFoldersByName("explanation-" + bot_name).next()
                                               .getFoldersByName("explanation").next().getName())
    var training_phase = iter_to_list(bot_files.getFoldersByName("explanation-" + bot_name).next()
                                               .getFoldersByName("explanation").next()
                                               .getFiles());
    training_phase = training_phase.slice(training_phase.length - NUM_TRAINING, training_phase.length);
  } else {
    Logger.log("control" + index.toString());
    Logger.log(bot_name);
    var training_phase = iter_to_list(bot_files.getFoldersByName("explanation-" + bot_name).next()
                                               .getFoldersByName("original").next()
                                               .getFiles())
                                               .slice(0, NUM_TRAINING);
  }
  Logger.log("Got training")
  // Test against all of the others.  We could consider doing fewer comparisons.
  var real_cont_list = iter_to_list(bot_files.getFoldersByName("explanation-" + bot_name).next()
                                        .getFoldersByName("original-cont").next()
                                        .getFiles())
                                        .slice(0, NUM_COMPARISONS * num_other_bots);
  Logger.log("Got real cont")
  // Random indices to use.  Determine whether the real or fake continuation should be on top
  var real_index = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1];
  var video1 = [];
  var video2 = [];
  var random_index = 0;
  // Loop through the other policies choosing trajectories to compare the original to.
  for (var i = 0; i < NUM_COMPARISONS; i++) {
    var expl_bots = bot_files.getFolders();
    while (expl_bots.hasNext()) {
      var expl_bot_files = expl_bots.next();
      // Don't continue the trajectory with the same policy we started with.
      if (expl_bot_files.getName() != "explanation-" + bot_name) {
        var real_cont = real_cont_list[i];
        var fake_cont = iter_to_list(expl_bot_files.getFoldersByName("explanation-cont").next()
                                                   .getFiles())[i];
        if (real_index[i] == 0) {
          video1.push(real_cont);
          video2.push(fake_cont);
        } else {
          video1.push(fake_cont);
          video2.push(real_cont);
        }
        i++;
      }
    }
  }
  Logger.log("Got cont")
  
  //DriveApp.getFileById('1YbrTohQqJ16yZYeFecxumFejmeYw5FEA') Sample video we can use for testing
  var testing_phase = {
    "video1": video1,
    "video2": video2,
  };
  
  var data = {
    "instruction_phase": explanation_videos,
    "training_phase": training_phase,
    "testing_phase": testing_phase,
  }
  generate_survey(data, counterfactual_version, index);
}
