// Task name
const TASK_NAME = 'GRIDWORLD'
// Number of train/test phase iterations
const NUM_TRAINING_PHASES = 3;
// Number of videos shown each training phase
const NUM_VIDEOS_PER_TRAIN = 2;
// Number of questions asked each testing phase
const NUM_QUESTIONS_PER_TEST = 1
// Number of multiple choice options in each testing phase questions
const NUM_OPTIONS_PER_TEST = 3
// Should we use the comparison task?
const USE_COMPARISON = true;
// Should we use the final position task?
const USE_FINAL = false; // TODO: change back
// Name of the policy the user is trying to guess
const POLICY = "WHATEVER";
// Name of the alt policies we use in the evaluation tasks
const COMPARISON_POLICIES = ["SOMETHING"];


// DIRECTORY STRUCTURE
// - GenerateForms-v2
// - videos_TASK_NAME/
//      - train/
//            - IDXXX/ (there will be multiple of these)
//                  - rollout_video.gif
//                  - explanation_video.gif
//      - test_comparison/
//.           - IDXXX/ (multiple of these)
//                  - POLICYNAMEX_comparison.gif (multiple of these)
//      - test_final_position/
//.           - IDXXX/ (multiple of these)
//                  - starting_video.gif
//                  - POLICYNAMEX_final_position.png (multiple of these)
//. - tasks/
//.     TASK_NAME/
//         miscellaneous videos/text for each task

// Note that we EITHER need test_comparison/ or test_final_positions/


//What we need
//- (1) Standard videos, train dist (main bot)
//- (2) Explanation Vides, train dist (main bot).  Optionally paired with (1)
//- (3) Standard videos, test dist (main bot)
//- (4) Standard videos, test dist (all other bots). Optionally paired with (3)
//- (5) Pre-video, test dist, (main bot)
//- (6) Final pos, test dist, (all other bots). Must be paired with (5).




function generate_all_surveys() {

  var curr_folder = get_curr_folder().getFoldersByName('videos' + TASK_NAME).next();
  var train_folder = curr_folder.getFoldersByName('train').next();
  var train_runs = train_folder.getFolders();
//  var start_index = 7;
//  var expert_trajs = iter_to_list(files).slice(start_index, start_index + NUM_EXAMPLE_EXPERT);

//  var example_suboptimal_name = 'DoubleForwardBot';
//  var suboptimal_traj_folder = get_curr_folder().getFoldersByName(example_suboptimal_name).next();
//  var files = suboptimal_traj_folder.getFiles();
//  var suboptimal_trajs = iter_to_list(files).slice(0, NUM_EXAMPLE_SUBOPTIMAL);
//
//  var suboptimal_trajs = [DriveApp.getFileById('1ma6yGghiNLq_T3lkoxAFxfStGVHouAXf'), DriveApp.getFileById('1mmPXRu7331j4SQBQSJxBvrj31Y6q0lX0')];
//
//  var counterfactual_traj_folder = expert_traj_folder.getFoldersByName('explanation-Bot').next()
//                                                     .getFoldersByName('explanation').next();
//  var files = counterfactual_traj_folder.getFiles();
//  var counterfactual_trajs = iter_to_list(files).slice(0, NUM_EXAMPLE_COUNTERFACTUAL);



  // TODO: make a task-specific list for each of these
  var explanation_videos = {};

//  var explanation_videos = {
//    "example_expert": expert_trajs,
//    "example_suboptimal": suboptimal_trajs,
//    "example_counterfactual": counterfactual_trajs,
//  };


  var index = 0;
  // Control surveys
  setup_survey(train_runs, COMPARISON_POLICIES[i], false, 0, explanation_videos);
  // Experimental surveys
  setup_survey(train_runs, COMPARISON_POLICIES[i], true, 1, explanation_videos);
}


function setup_survey(train_runs, comparison_policy, counterfactual_version, index, explanation_videos) {
  /** Generate a single survey. */
  if (counterfactual_version) {
    Logger.log("counterfactual" + index.toString())
    var training_phase = [];
    while (iterator.hasNext()) {
      training_phase.push(iterator.next().getFilesByName("explanation_video.gif").next());
    }
    training_phase = training_phase.slice(training_phase.length - NUM_TRAINING, training_phase.length);
  } else {
    Logger.log("control" + index.toString());
    Logger.log(bot_name);
    var training_phase = [];
    while (iterator.hasNext()) {
      training_phase.push(iterator.next().getFilesByName("rollout_video.gif").next())
    }
    training_phase = training_phase.slice(0, NUM_TRAINING);
  }
  Logger.log("Got training")
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
  if (USE_COMPARISON) {
  }
  if (USE_FINAL) {
  }


  // Loop through the other policies choosing trajectories to compare the original to.
  for (var i = 0; i < COMPARISON_POLICIES.length; i++) {
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
  var task = {
    "instructions": [],

  }
  generate_survey(task, data, counterfactual_version, index);
}
