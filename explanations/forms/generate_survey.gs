function generate_survey(data, counterfactual_version, id) {
  
  var form_name = "Explaining Robot Behavior: Version " + id.toString();
   // Create Form   
   var form = FormApp.create(form_name)  
                     .setTitle(form_name)
                     .setDescription('Thank you for participating in our user study. '
                                      + 'We are exploring tools to help users understand robotic behavior. '
                                     )
                     .setConfirmationMessage('Thanks for responding!')
                     .setAcceptingResponses(true);  
  
  // Describe the task
  add_task_description(form, data, counterfactual_version);
  
  // Add one experiment
  // If we ever wanted to test a single user on multiple, we could add more here.
  experiment(form, data, counterfactual_version);
  
  // Add summary questions afterwards
  form.addPageBreakItem()
      .setTitle('Subjective Questions')
      .setHelpText('Please do not go back to the previous page while answering these questions.');
  
  if (counterfactual_version) {
    form.addScaleItem()
        .setLabels("Confusing/Unhelpful", "Very Helpful")
        .setTitle("How much did the counterfactual trajectories help you understand how the robot generally acts?").setRequired(true);
    form.addParagraphTextItem().setTitle("Explain.");
    
    form.addScaleItem()
        .setLabels("Confusing/Unhelpful", "Very Helpful")
        .setTitle("How much did the counterfactual trajectories help you understand why the robot "
                                        +"acted suboptimally in particular scenarios?").setRequired(true);
    form.addParagraphTextItem().setTitle("Explain.").setRequired(true);
  }
  
  form.addScaleItem()
        .setLabels("Confusing/Unhelpful", "Very Helpful")
        .setTitle("Imagine there was a tool available which allowed you to pause a video of a robot in action, "
                                      +"control the robot yourself to navigate it to any part of the grid, "
                                      +"and then let the robot finish the task. "
                                      +"How useful would this tool be to help you understand the behavior?").setRequired(true);
  form.addParagraphTextItem().setTitle("Explain.").setRequired(true);
  
  
  form.addParagraphTextItem().setTitle("Are there any other types of videos, explanations, or tools you would like to see "
                                      +"which would help you understand what the robot does and why?").setRequired(true);
  var textValidation = FormApp.createTextValidation()
  .setHelpText("Input was not a nonnegative number.")
  .requireNumberGreaterThan(-0.1)
  .build();
  form.addTextItem()
      .setTitle("We showed you 7 trajectories to help you understand the robot's behavior. "
               +"How many trajectories do you think you would need to understand the behavior? (may be higher or lower)")
      .setValidation(textValidation).setRequired(true);
  
  form.addScaleItem().setLabels("Too Short", "Too Long").setRequired(true).setTitle("How was the length of this survey?");
  form.addParagraphTextItem().setTitle("Additional Comments");
  
  
  // By default, the form gets created in the home folder.  Instead, move it to the XRL folder.
  moveToCurrentFolder(form);
}
