using UnityEngine;
using System.Xml;
using System.IO;
using System.Collections.Generic;
using System;
using System.Diagnostics;
using UnityEngine.Assertions;
using System.Linq;
using System.Text.RegularExpressions;

public class LoadLevelSchema : ABSingleton<LoadLevelSchema>{
    public string[] activeLevelFolders{get;set;}
    public int timeLimit;
    public int interactionLimit; 

    public int attemptsLimitPerLevel;
    public int timeLeft {get;set;}
    public int interactionLeft {get;set;}
    public int noveltyLikelihoodRequestFreq{get;set;}

    public bool initializationDone;
    public bool measure_in_training;
    public bool measure_in_testing;
    public bool isCheckPoint;
    public bool isNewTrial;
    public bool doNotUpdateLevels;
    public bool firstLoad;
    public bool rerunTrial;
    public bool resumeTraining;
    public bool needResetTimer;
    public bool noveltyLikelihoodReportRequest;
    public bool isTesting;
    public bool isSequence;

    public bool informedOnly;
    public bool uninformedOnly;
    public bool loadTrialList;//load a trial list to specify which trials to use for evaluation

    //if true: all evaluation trials are loaded and evalation will end after next level set update
    public bool evaluationTerminating; 

    public int currentTrialIndex;

    public int prevLevelIndex;

    public int tempReachedLevelIndex;//the index of level reached at each run

    public int nInteractions;

    public int nOverallInteractions;
    public List<TrialInfo> trials;
    public string[] currentXmlFiles {get;set;}
    public string[] currentLevelPaths{get;set;}

    //flags/error codes
    //  report detection likelihood before next interaction
    //  checkpoint reaction
    public Stopwatch overallStopwatch = new Stopwatch();

    public Stopwatch checkpointWatch = new Stopwatch();
    public bool devMode {get; private set;}

    private string configXmlPath;
    private bool generateConfigFile;
    private string trialListPath;
    private List<int> trialListFromFile;
    private int specificTrialIndex;

    //the start and end trial index to be used in the evaluation
    int startTrialIndex, endTrialIndex;
    bool trialIndicesSpecified;
    void Awake(){
        initializationDone = false;
        generateConfigFile = true;
        trialIndicesSpecified = false;
        configXmlPath = ABConstants.STREAMINGASSETS_FOLDER + "/config.xml";
        informedOnly = false;
        uninformedOnly = false;
        loadTrialList = false;
        devMode = false;
        trialListPath = "./trial_list";
        specificTrialIndex = 0;
    }

    void Start(){

 //     DontDestroyOnLoad(this.gameObject);
        UnityEngine.Debug.Log("load level schema started");
        evaluationTerminating = false;
        isCheckPoint = true;//set to true for the initial testing/training state to be sent
        needResetTimer = true;
        isNewTrial = true;
        doNotUpdateLevels = false;
        firstLoad = true;
        resumeTraining = false;
        noveltyLikelihoodReportRequest = false;

        prevLevelIndex = 0;
        attemptsLimitPerLevel = 0;//0 means no limit

        nInteractions = 0;
        nOverallInteractions = 0;
        trials = new List<TrialInfo>();
        
        string[] args = System.Environment.GetCommandLineArgs ();
        for (int i = 0; i < args.Length; i++) {
            UnityEngine.Debug.Log ("ARG " + i + ": " + args [i]);
            if (args [i] == "--configpath") {
                configXmlPath = args [i + 1];
                generateConfigFile = false;
            }

            if(args[i]=="--trial"){
                string[] trialIndecies = args[i+1].Split('-');
                
                if(trialIndecies.Length > 1){
                    startTrialIndex = Int32.Parse(trialIndecies[0]);
                    endTrialIndex = Int32.Parse(trialIndecies[1]);
                }
                else{
                    startTrialIndex = Int32.Parse(trialIndecies[0]);
                    endTrialIndex = Int32.Parse(trialIndecies[0]);
                }
                
                //swap if start is larger than end
                if(startTrialIndex > endTrialIndex){
                    int tempIndex = startTrialIndex;
                    startTrialIndex = endTrialIndex;
                    endTrialIndex = tempIndex;
                }
                this.trialIndicesSpecified = true;
            }

            if(args[i]=="--informed-only"){
                this.informedOnly = true;
            }
            if(args[i]=="--uninformed-only"){
                this.uninformedOnly = true;
            }

            //overwrite the default trial list path
            if(args[i]=="--trial-list-path"){
                this.trialListPath = args[i+1];
            }

            //if load a trial list to specify which trials to use for evaluation
            if(args[i]=="--load-trial-list"){
                this.loadTrialList = true;
            }
            if(args[i]=="--dev"){
                this.devMode = true;
            }
        }


        if(this.loadTrialList){
            this.trialListFromFile = loadTrials(this.trialListPath);
        }
        //generate a new config file first
        //if no config path is given
        if(generateConfigFile){
            //updateConfig();
        }
        
        readTestConfig(configXmlPath);
        if(!trialIndicesSpecified){
            startTrialIndex = 0;
            if(!loadTrialList){
                endTrialIndex = trials.Count-1;
            }
            else{
                endTrialIndex = this.trialListFromFile.Count-1;
            }
        }


        //repeat the last trial if the specified indicies are greater than the number of trials
        if(!loadTrialList){
            if(startTrialIndex > trials.Count-1){
                startTrialIndex = trials.Count-1;
            }
            if(endTrialIndex > trials.Count-1){
                endTrialIndex = trials.Count-1;
            }

        }
        else{//load trials from list
            if(startTrialIndex > this.trialListFromFile.Count-1){
                startTrialIndex = this.trialListFromFile.Count-1;
            }
            if(endTrialIndex > this.trialListFromFile.Count-1){
                endTrialIndex = this.trialListFromFile.Count-1;
            }
        }

        if(!loadTrialList){
            currentTrialIndex = startTrialIndex;
        }
        else{
            this.specificTrialIndex = startTrialIndex;
            currentTrialIndex = this.trialListFromFile[this.specificTrialIndex];
        }
        
        //initialize, will be updated in function updateLevelSet() 
        timeLeft = 0;
        interactionLeft = 0;

        updateLevelSet();

        //init save path of evaluation data
        //EvaluationHandler.Instance.savePathUpdate(configXmlPath);
        
        //do not do reset and start timer here
        //instead do them when receiving readyfornewset command
        //stopwatch.Reset();
        //stopwatch.Start();

        //InvokeRepeating("TestStatusUpdate", 1.0f, 1.0f);
    }

    /*
        load specific trials to use from a trial list file
        should only be used when  
    */
    private List<int> loadTrials(string path){
        List<int> trials = new List<int>();
        string line;  
        
        // Read the file and display it line by line.  
        System.IO.StreamReader file =
            new System.IO.StreamReader(path);  
        while((line = file.ReadLine()) != null){  
            try{
                int trialIndexFromLine = Int32.Parse(line);
                trials.Add(trialIndexFromLine);
            }
            catch (FormatException){
                UnityEngine.Debug.LogWarning($"Unable to parse '{line}'");
            }
        }
        return trials;                
    }

    /*
    read test settings from the xml file
    */
    public void readTestConfig(string configXmlPath){
        
        string configXml;
        try{
            configXml = File.ReadAllText(configXmlPath);
        }
        catch (FileNotFoundException){  
            UnityEngine.Debug.LogWarning("input config file not found, using default one instead...");
            configXml = File.ReadAllText(ABConstants.STREAMINGASSETS_FOLDER + "/config.xml");
        }
        XmlReaderSettings readerSettings = new XmlReaderSettings();
        readerSettings.IgnoreComments = true;
//        readerSettings.IgnoreProcessingInstructions = true;
        readerSettings.IgnoreWhitespace = true;
        XmlReader reader = XmlReader.Create(new StringReader(configXml),readerSettings);
        reader.ReadToFollowing("evaluation");

        reader.ReadToFollowing("novelty_detection_measurement");
        reader.MoveToAttribute("step");
        noveltyLikelihoodRequestFreq = int.Parse(reader.Value);

        reader.MoveToAttribute("measure_in_training");
        string measure_in_training_string = reader.Value;
        if(measure_in_training_string =="True"){
            this.measure_in_training = true;
        }
        else{
            this.measure_in_training = false;
        }
        reader.MoveToAttribute("measure_in_testing");
        string measure_in_testing_string = reader.Value;
        if(measure_in_testing_string =="True"){
            this.measure_in_testing = true;
        }
        else{
            this.measure_in_testing = false;
        }


        reader.ReadToFollowing("trials");
//        reader.Read();
        //read trials
        
        while(reader.Read()){
            if(reader.NodeType == XmlNodeType.Comment){
                reader.Read();
                continue;
            }
            
            string nodeName = reader.LocalName;
            if (nodeName == "trials")
                break;
            
            TrialInfo trial = new TrialInfo();

            reader.MoveToAttribute("id");
            trial.id = int.Parse(reader.Value);
            reader.MoveToAttribute("number_of_executions");
            int number_of_executions = int.Parse(reader.Value);
            if(reader.GetAttribute("checkpoint_time_limit")!=null){
                int checkPointTimeLimit = int.Parse(reader.GetAttribute("checkpoint_time_limit"));
                if(checkPointTimeLimit > 0){
                    trial.hasCheckpointTimeLimit = true;
                    trial.checkpointTimeLimit = checkPointTimeLimit;
                }
                else{
                    trial.hasCheckpointTimeLimit = false;
                }
            }
            if(reader.GetAttribute("checkpoint_interaction_limit")!=null){
                int checkPointInteractionLimit = int.Parse(reader.GetAttribute("checkpoint_interaction_limit"));

                if(checkPointInteractionLimit >0){
                    trial.hasCheckpointTimeLimit = true;
                    trial.checkpointInteractionLimit = checkPointInteractionLimit;
                }
                else{
                    trial.hasCheckpointInteractionLimit = false;
                }
            }

            if(reader.GetAttribute("notify_novelty")!=null){
                string notifyNoveltyString = reader.GetAttribute("notify_novelty");
                if(notifyNoveltyString == "True"){
                    trial.notifyNovelty = true;
                }
                else{
                    trial.notifyNovelty = false;
                }
            }

            //reader.Read();
            //read game_level_set
            while(reader.Read()){
                if(reader.NodeType == XmlNodeType.Comment){
                    reader.Read();
                    continue;
                }
                nodeName = reader.LocalName;
                if (nodeName == "trial")
                    break;
                GameLevelSetInfo gameLevelSet;
                reader.MoveToAttribute("mode");
                string mode = reader.Value;
                switch (mode){
                    case "training": 
                        gameLevelSet = new TrainingSet();
                        gameLevelSet.mode=GameLevelSetInfo.Mode.TRAINING;
                        break;
                    case "test":
                        gameLevelSet = new TestSet();
                        gameLevelSet.mode=GameLevelSetInfo.Mode.TEST;
                        break;
                    default:
                        gameLevelSet = new GameLevelSetInfo();
                        gameLevelSet.mode=GameLevelSetInfo.Mode.UNKNOWN;
                        break;
                }

                //only one training set is allowed
                if(trial.trainingSet!=null&&gameLevelSet.mode==GameLevelSetInfo.Mode.TRAINING){
                    continue;
                }

                reader.MoveToAttribute("time_limit");
                gameLevelSet.timeLimit = int.Parse(reader.Value);
                reader.MoveToAttribute("total_interaction_limit");
                gameLevelSet.interactionLimit = int.Parse(reader.Value);
                reader.MoveToAttribute("allow_level_selection");
                string allowLevelSelectionString = reader.Value;
                reader.MoveToAttribute("attempt_limit_per_level");
                if(allowLevelSelectionString=="True"){
                    gameLevelSet.isSequence = false;
                }
                else{
                    gameLevelSet.isSequence = true;
                }
                int attemptsPerLevel = int.Parse(reader.Value);
                gameLevelSet.levelAttemptLimit = attemptsPerLevel;
                if(attemptsPerLevel>0){
                    gameLevelSet.hasLevelAttepmtLimit = true;
                }
                // read level groups with the same novelty level and type in a level set
                List<string> xmlStrings = new List<string>();
                List<string> levelPaths = new List<string>();
                while(reader.Read()){
                    if(reader.NodeType == XmlNodeType.Comment){
                        reader.Read();
                        continue;
                    }
                    nodeName = reader.LocalName;
                    if (nodeName == "game_level_set")
                        break;
//                    reader.MoveToAttribute("novelty_level");
//                    string noveltyLevel = reader.Value;
//                    reader.MoveToAttribute("type");
//                    string noveltyType = reader.Value;
                    reader.MoveToAttribute("level_path");
                    string levelPath = reader.Value;
                    //reader.MoveToAttribute("end");
                    //int endIndex = int.Parse(reader.Value);
                    //string levelParentPath = Application.dataPath + ABConstants.CUSTOM_LEVELS_FOLDER + "/" + noveltyLevel+"/" +noveltyType;
                    //string levelPath = levelParentPath +"/Levels";
                    //string[] levelFiles = Directory.GetFiles(levelPath, "*.xml").OrderBy(f => Regex.Replace(f, @"\d+", n => n.Value.PadLeft(4, '0'))).ToArray();
                                        
                    //levelPaths.AddRange(levelFiles);
//                    if(endIndex > levelFiles.Length-1){
//                        endIndex = levelFiles.Length-1;
//                    }
                    //if(startIndex < 0){
                    //    startIndex = 0;
                    //}
                    levelPaths.Add(levelPath);
//                    for (int i = startIndex; i <= endIndex; i++){
//                        levelPaths.Add(levelFiles[i]);
//                        UnityEngine.Debug.Log("index " + i);
//                        UnityEngine.Debug.Log("path " + levelFiles[i]);
//                    }
                }
                if(levelPaths==null){
                    continue;
                }
                gameLevelSet.levelPaths = levelPaths;
                gameLevelSet.resetAvailableLevelList();

                if(typeof(TrainingSet).IsInstanceOfType(gameLevelSet)){
                    trial.trainingSet = (TrainingSet)gameLevelSet;
                }
                else if(typeof(TestSet).IsInstanceOfType(gameLevelSet)){
                    trial.testLevelSets.Add((TestSet)gameLevelSet);                                        
                }
                else{
                    //ignore other types of sets for now
                    //TODO may extend later
                }
            }

            //make sure the training set is set
            Assert.IsNotNull(trial.trainingSet);

            if(uninformedOnly && trial.notifyNovelty){
                continue;
            }

            if(informedOnly && !trial.notifyNovelty){
                continue;
            }
            //add the trial based on the number of executions of the trial            
            trials.Add(trial);

        }
    }
    
    public void updateConfig(){
        //if #evaluation times not reached
        evaluationTerminating = false;
        isNewTrial = true;
        var configGenerator = new ConfigGenerator();
        configGenerator.generateTestConfigFile();
    }

    public void updateLevelSet(){

        TrialInfo trial = trials[currentTrialIndex];
        GameLevelSetInfo gameSet;
        if(isNewTrial){       

            nInteractions = 0;
            nOverallInteractions = 0;
            overallStopwatch.Reset();
            overallStopwatch.Start();
            checkpointWatch.Reset();
            checkpointWatch.Start();

            if(rerunTrial){
                trial.reset();
                trial.currentExecutionIndex++;
                gameSet = trial.trainingSet;
                //reset all timers and interactions
                //EvaluationHandler.Instance.UpdateEvaluationFileInfo(currentTrialIndex.ToString(),trial.currentExecutionIndex.ToString(),"0","training");
            }
            else if(!initializationDone){
                trial = trials[currentTrialIndex];
                gameSet = trial.trainingSet;
                //EvaluationHandler.Instance.UpdateEvaluationFileInfo(currentTrialIndex.ToString(),trial.currentExecutionIndex.ToString(),"0","training");                
            }
            else{
                if(!this.loadTrialList){
                    currentTrialIndex++;
                }
                else{//if use arbitary trial indices from a file  
                    this.specificTrialIndex++;
                    currentTrialIndex = this.trialListFromFile[specificTrialIndex];
                }
                trial = trials[currentTrialIndex];
                gameSet = trial.trainingSet;
                //EvaluationHandler.Instance.UpdateEvaluationFileInfo(currentTrialIndex.ToString(),trial.currentExecutionIndex.ToString(),"0","training");
            }
            updateGameLevelSet(gameSet);
            isNewTrial = false;
            rerunTrial = false;

        }
        else if(isCheckPoint){
            checkpointWatch.Reset();
            checkpointWatch.Start();
            nInteractions = 0;

            if(isTesting){
                trial.currentTestSetIndex++;
            }
            else{
                trial.currentTestSetIndex = 0;
            }
            //continue testing
            if(trial.currentTestSetIndex <= trial.testLevelSets.Count-1){
                gameSet = trial.testLevelSets[trial.currentTestSetIndex];
                updateGameLevelSet(gameSet);
            }

            isCheckPoint = false;
        }

        else if(resumeTraining){
            checkpointWatch.Reset();
            checkpointWatch.Start();
            nInteractions = 0;

            gameSet = trial.trainingSet;
            updateGameLevelSet(gameSet);
            resumeTraining = false;
        }

        if(!initializationDone){
            initializationDone = true;
        }

    }

    private void updateGameLevelSet(GameLevelSetInfo gameSet){
        currentXmlFiles = gameSet.getLevelSetXmlData().ToArray();
        currentLevelPaths = gameSet.levelPaths.ToArray();
        this.timeLimit = gameSet.timeLimit;
        this.interactionLimit = gameSet.interactionLimit;
        this.timeLeft = gameSet.timeLimit;
        this.interactionLeft = gameSet.interactionLimit;

        if(gameSet.mode==GameLevelSetInfo.Mode.TEST){
            isTesting = true;
        }
        else{
            isTesting = false;

        }
        isSequence = gameSet.isSequence;
        attemptsLimitPerLevel = gameSet.levelAttemptLimit; 
        
        LevelList.Instance.RefreshLevelsFromSource(this.currentXmlFiles);
        if(!resumeTraining){
            needResetTimer = true;
        }

    }

    public void updateLikelihoodRequestFlag(){
        UnityEngine.Debug.Log("try to update request likelihood report flag");
        TrialInfo trial = trials[currentTrialIndex];
        
        if(noveltyLikelihoodRequestFreq > 0)
        {
            if((measure_in_training&&(!isTesting))||(measure_in_testing&&isTesting)){
                UnityEngine.Debug.Log("request likelihood report flag is allowed in the current mode, checking if it should be set to true");
                // do not request if the novelty is already nofified
                if(nInteractions%noveltyLikelihoodRequestFreq == 0 && nInteractions!=0 && !trial.notifyNovelty){
                    noveltyLikelihoodReportRequest = true;
                    UnityEngine.Debug.Log("request likelihood report flag is set to true");
                }
            }
        }
    }    

    void Update()
    {
        if(!isCheckPoint && !resumeTraining && !isNewTrial && !evaluationTerminating){
            TrialInfo trial = trials[currentTrialIndex];
            TimeSpan ts, tsCheckPoint;
            int secondElapsed, secondElapsedCheckPoint;
            tsCheckPoint = checkpointWatch.Elapsed;
            secondElapsedCheckPoint = tsCheckPoint.Hours*3600+tsCheckPoint.Minutes*60 + tsCheckPoint.Seconds; 

            if(!isTesting){
                ts = overallStopwatch.Elapsed;
                secondElapsed = ts.Hours*3600+ts.Minutes*60 + ts.Seconds;
                
                if((trial.trainingSet.timeLimit>0 && secondElapsed >= trial.trainingSet.timeLimit)||
                (trial.trainingSet.interactionLimit > 0 && nOverallInteractions >= trial.trainingSet.interactionLimit)||
                (trial.isTrainingFinished())){//any termination conditions meet

                    trial.trainingSet.trainingTerminated = true;
                
                    if(trial.testLevelSets.Count>0){
                        isCheckPoint = true;
                    }
                    else{
                        if(trial.currentExecutionIndex < trial.nExecutions){
                            isNewTrial = true;
                            rerunTrial = true;                                    
                        }
                        else if(!this.loadTrialList&&(currentTrialIndex < endTrialIndex)){
                            isNewTrial = true;
                        } 

                        else if(this.loadTrialList&&(this.specificTrialIndex < endTrialIndex)){
                            isNewTrial = true;
                        }        
                        else{
                            evaluationTerminating = true;
                        }           
                    }
                }
                else if((trial.checkpointTimeLimit > 0 && secondElapsedCheckPoint >= trial.checkpointTimeLimit)||
                (trial.checkpointInteractionLimit > 0 && nInteractions >= trial.checkpointInteractionLimit)){
                    if(trial.testLevelSets.Count>0){
                        isCheckPoint = true;
                        prevLevelIndex = LevelList.Instance.CurrentIndex;                    
                    }
                    else{//no test set in the trial, ignore check point
                        checkpointWatch.Reset();
                        checkpointWatch.Start();                        
                    }
                
                }

            }
            else{//current state is testing
                TestSet testSet = trial.testLevelSets[trial.currentTestSetIndex]; 
                if((testSet.timeLimit>0 && secondElapsedCheckPoint >= testSet.timeLimit)||
                (testSet.interactionLimit > 0 && nInteractions >= testSet.interactionLimit)||
                (trial.isTestSetFinished())){
                    if(trial.currentTestSetIndex >= trial.testLevelSets.Count-1){
                        if(trial.trainingSet.trainingTerminated){
                            if(trial.currentExecutionIndex < trial.nExecutions){
                                isNewTrial = true;
                                rerunTrial = true;                                    
                            }
                            else if(!this.loadTrialList&&(currentTrialIndex < endTrialIndex)){
                                isNewTrial = true;
                            } 

                            else if(this.loadTrialList&&(this.specificTrialIndex < endTrialIndex)){
                                isNewTrial = true;
                            }       
                            else{
                                evaluationTerminating = true;
                            }
                        }
                        else{
                            resumeTraining = true;
                        }
                    }
                    else{//continue with next test set
                        isCheckPoint = true;
                    }
                }
            }
        }            
    }
}