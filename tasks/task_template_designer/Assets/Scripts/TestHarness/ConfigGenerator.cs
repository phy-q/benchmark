using UnityEngine;
using System.IO;
using System.Xml;
using System.Collections.Generic;
using System;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;


class TrainingEntry{

    public int[] timeLimit;
    public int interactionLimit;
    public int[] nTrainingLevels;
    public int attemptPerLevel;
    public bool allowLevelSelection;
    public string noveltyLevel;
    public string noveltyType;
    public double noveltyAbsencePortion;
    public double noDifferencePortion;
    public double noveltyPortion;


    public TrainingEntry(){
        noveltyAbsencePortion = 0;
        noDifferencePortion = 0;
        noveltyPortion = 1;
        allowLevelSelection = false;
    }
    
}
class TestSetEntry{
    public int timeLimit;
    public int interactionLimit;
    public int attemptPerLevel;
    public bool allowLevelSelection;
    public string noveltyLevel;
    public string noveltyType;
    public int nLevels;
    public int testSetPercentage;
}

class Trial{
    public int nTrials;
    public int trialRepeats;
    public int trialCount;
    
    public bool testSetOrder;
    public int avgTrainingTimeLimitPerLevel;
    public int[] overallTrainingTimeLimit;
    public int trainingInteractionLimit;
    public int trainingAttemptPerLevel;

    public bool trainingAllowLevelSelection;

    public bool notifyNovelty;

    public bool bothNotifyNovelty;//generate the same trial for both both notify and non notify
    public int checkPointInteractionLimit;
    public int checkPointTimeLimit;

    public List<int> testSetPercentage;

    public List<TrainingEntry> trainingEntryArray;
    public List<TestSetEntry> testEntryArray;

    //<noveltyLevel, dict<noveltyType, levelPaths>>

    public Trial(){
        checkPointInteractionLimit = 0;
        checkPointTimeLimit = 0;
        testSetOrder = true;
        trainingEntryArray = new List<TrainingEntry>();
        testEntryArray = new List<TestSetEntry>();
        notifyNovelty = false;
        bothNotifyNovelty = false;

    }

}


public class ConfigGenerator{
    private System.Random rnd = new System.Random();  
    List<Trial> trials;
    Dictionary<int, Dictionary<int,string[]>> noveltyInfoDict;
    int noveltyDetectionReportStep;
    bool testMeasureNoveltyLikelihood;
    bool trainingMeasureNoveltyLikelihood;
    int nEvaluations;

    public ConfigGenerator(){
        trials = new List<Trial>();
        noveltyInfoDict = new Dictionary<int, Dictionary<int,string[]>>();
    }
 
    private void Shuffle<T>(IList<T> list)  
    {  
        int n = list.Count;  
        while (n > 1) {  
            n--;  
            int k = rnd.Next(n + 1);  
            T value = list[k];  
            list[k] = list[n];  
            list[n] = value;  
        }  
    }

    private void getAllLevelInfo(){
        string parentPath = Application.dataPath + ABConstants.CUSTOM_LEVELS_FOLDER;    
        //sorted by name
		string[] noveltyLevelDirs = Directory.GetDirectories(parentPath, "novelty_level_*", SearchOption.TopDirectoryOnly).OrderBy(f => Regex.Replace(f, @"\d+", n => n.Value.PadLeft(4, '0'))).ToArray();
		for (int i=0; i< noveltyLevelDirs.Length;i++){
            string levelDir = noveltyLevelDirs[i];
            string[] segs = levelDir.Split('_');
            int level = int.Parse(segs[segs.Length-1]);
            //sorted by name
            string[] noveltyTypeDirs = Directory.GetDirectories(levelDir, "type*", SearchOption.TopDirectoryOnly).OrderBy(f => Regex.Replace(f, @"\d+", n => n.Value.PadLeft(4, '0'))).ToArray();
            Dictionary<int,string[]> typeDict = new Dictionary<int, string[]>();

//            Debug.Log(levelDir);
            for (int j = 0; j <  noveltyTypeDirs.Length; j++){
                string typeDir = noveltyTypeDirs[j];
                string[] levelFiles = Directory.GetFiles(typeDir+"/Levels/", "*.xml");
                string[] segsType = typeDir.Split('e');
                int type = int.Parse(segsType[segsType.Length-1]);
                typeDict.Add(type,levelFiles);
			}
            noveltyInfoDict.Add(level,typeDict);
		}    
    }

    public void generateTestConfigFile(){
        string configMetaPath = ABConstants.STREAMINGASSETS_FOLDER + "/configMeta.xml"; 
        getAllLevelInfo();
        readTestGenerationConfig(configMetaPath); 
    }

    private void writeMetaConfigFile(string metaConfigPath){
        StringBuilder output = new StringBuilder();
		XmlWriterSettings ws = new XmlWriterSettings();
		ws.Indent = true;

        double[] noveltyDiffProportion = {0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05};
        double[] noDiffRelativeProportion = {0, 0.2, 0.6, 0.8, 1};
        double[] noNoveltyRelativeProportion = {1, 0.8, 0.4, 0.2, 0};
        
        //int nonNovelLowerLimit = 10;
        // int nonNovelUpperLimit = 50;
        // int noveltyCount = 100;
        
        string step = "1";
        string measureInTraining = "true";
        string measureInTesting = "true";

		using (XmlWriter writer = XmlWriter.Create(output, ws))
		{
            writer.WriteStartElement("config_generation");

            writer.WriteStartElement("novelty_detection_measurement");
            writer.WriteAttributeString("step", step );
            writer.WriteAttributeString("measure_in_training",measureInTraining);
            writer.WriteAttributeString("measure_in_testing", measureInTesting);
            writer.WriteEndElement();//novelty_detection_measurement
            
            writer.WriteStartElement("trials");
            for(int noveltyLevel = 1; noveltyLevel < 4; noveltyLevel ++){
                for(int noveltyType = 1; noveltyType < 6; noveltyType ++){
                    foreach(double noveltyDiff in noveltyDiffProportion){
                        double remainingProportion = 1 - noveltyDiff;
                        for(int i = 0; i < noDiffRelativeProportion.Length; i++){
                            
                            double noDiff = noDiffRelativeProportion[i]*remainingProportion;
                            double noNovelty = noNoveltyRelativeProportion[i]*remainingProportion;
                            writer.WriteStartElement("trial");
                            writer.WriteAttributeString("notify_novelty","both");
                            
                            writer.WriteStartElement("count");
                            writer.WriteString("2");
                            writer.WriteFullEndElement();//count
                            
                            writer.WriteStartElement("repeats");
                            writer.WriteString("1");
                            writer.WriteFullEndElement();//repeats

                            writer.WriteStartElement("checkpoint");
                            
                            writer.WriteStartElement("limit");
                            writer.WriteAttributeString("type","time");
                            writer.WriteString("200");
                            writer.WriteEndElement();//limit
                            writer.WriteStartElement("limit");
                            writer.WriteAttributeString("type","interaction");
                            writer.WriteString("200");
                            writer.WriteEndElement();//limit
                            
                            writer.WriteFullEndElement();//checkpoint

                            writer.WriteStartElement("training");

                            writer.WriteStartElement("limits");
                            writer.WriteStartElement("limit");
                            writer.WriteAttributeString("type","time");
                            writer.WriteString("120");
                            writer.WriteEndElement();//limit
                            writer.WriteStartElement("limit");
                            writer.WriteAttributeString("type","interaction");
                            writer.WriteString("6000");
                            writer.WriteEndElement();//limit
                            writer.WriteStartElement("limit");
                            writer.WriteAttributeString("type","attempt_per_level");
                            writer.WriteString("1");
                            writer.WriteEndElement();//limit
                            writer.WriteStartElement("limit");
                            writer.WriteAttributeString("type","allow_level_selection");
                            writer.WriteString("false");
                            writer.WriteEndElement();//limit
                            writer.WriteFullEndElement();//limits


                            writer.WriteStartElement("game_level_sets");
                            
                            writer.WriteStartElement("game_level_set");
                            writer.WriteAttributeString("novelty_level","0");
                            writer.WriteAttributeString("novelty_type","1");
                            writer.WriteAttributeString("amount","10-50");
                            writer.WriteAttributeString("novelty_absence_portion","1");
                            writer.WriteAttributeString("no_diff_portion","0");
                            writer.WriteEndElement();//game_level_set

                            writer.WriteStartElement("game_level_set");
                            writer.WriteAttributeString("novelty_level",noveltyLevel.ToString());
                            writer.WriteAttributeString("novelty_type",noveltyType.ToString());
                            writer.WriteAttributeString("amount","100");
                            writer.WriteAttributeString("novelty_absence_portion",noNovelty.ToString());
                            writer.WriteAttributeString("no_diff_portion",noDiff.ToString());
                            writer.WriteEndElement();//game_level_set

                            writer.WriteFullEndElement();//game_level_sets

                            writer.WriteFullEndElement();//training

                            writer.WriteStartElement("test");
                            writer.WriteAttributeString("ordered","true");
                            writer.WriteStartElement("game_level_sets");
                            writer.WriteFullEndElement();//game_level_sets
                            writer.WriteFullEndElement();//test
                            

                            writer.WriteFullEndElement();//trial

                        }
                    }
                }
            }


            writer.WriteFullEndElement();//trials
            
            writer.WriteFullEndElement();//config_generation

        }
        StreamWriter streamWriter;
        streamWriter = new StreamWriter(metaConfigPath);
		streamWriter.WriteLine(output.ToString());
		streamWriter.Close();
    }

    private void writeConfigFile(string configPath){
        StringBuilder output = new StringBuilder();
		XmlWriterSettings ws = new XmlWriterSettings();
		ws.Indent = true;

		using (XmlWriter writer = XmlWriter.Create(output, ws))
		{
            writer.WriteStartElement("evaluation");

            writer.WriteStartElement("novelty_detection_measurement");
            writer.WriteAttributeString("step", noveltyDetectionReportStep.ToString());
            writer.WriteAttributeString("measure_in_training",trainingMeasureNoveltyLikelihood.ToString());
            writer.WriteAttributeString("measure_in_testing", testMeasureNoveltyLikelihood.ToString());
            writer.WriteEndElement();
            
            writer.WriteStartElement("trials");

            int trialID = 0;
            foreach(Trial trial in trials){

                for(int i = 0; i < trial.trialCount; i++){
                    int repeatTrial = 1;
                    if(trial.bothNotifyNovelty){
                        repeatTrial = 2;
                        trial.notifyNovelty = false;//for both notify and not notify
                    }
                    List<string> allSelectedLevelList = new List<string>(); 
                    while(repeatTrial > 0){
                        repeatTrial--;

                        writer.WriteStartElement("trial");
                        writer.WriteAttributeString("id",trialID.ToString());
                        trialID++;
                        writer.WriteAttributeString("number_of_executions",trial.trialRepeats.ToString());
                        writer.WriteAttributeString("checkpoint_time_limit", trial.checkPointTimeLimit.ToString());
                        writer.WriteAttributeString("checkpoint_interaction_limit", trial.checkPointInteractionLimit.ToString());
                        writer.WriteAttributeString("notify_novelty", trial.notifyNovelty.ToString());
                        //one training set
                        writer.WriteStartElement("game_level_set");
                        writer.WriteAttributeString("mode", "training");
                        writer.WriteAttributeString("time_limit", trial.overallTrainingTimeLimit[i].ToString());
                        writer.WriteAttributeString("total_interaction_limit",trial.trainingInteractionLimit.ToString());
                        writer.WriteAttributeString("attempt_limit_per_level", trial.trainingAttemptPerLevel.ToString());
                        writer.WriteAttributeString("allow_level_selection",trial.trainingAllowLevelSelection.ToString());
                        //writer.WriteFullEndElement();

                        int nTotalTrainingLevels = 0;
                        var sampledDict = new Dictionary<(int, int), List<int>>();
                        List<string> prevSelectedLevelList = new List<string>(); 
                        if(repeatTrial >= 1 || !trial.bothNotifyNovelty){

                            foreach(var trainingEntry in trial.trainingEntryArray){

                                int trainingNoveltyLevel;
            /*                    if(trainingEntry.noveltyLevel=="*"){
                                    List<int> levels = new List<int>();
                                    levels.AddRange(noveltyInfoDict.Keys);
                                    Shuffle(levels);  
                                    trainingNoveltyLevel = levels[0];
                                }
                                else{*/
                                trainingNoveltyLevel = int.Parse(trainingEntry.noveltyLevel);
            //                    }

                                int trainingNoveltyType;
                                Dictionary<int,string[]> levelPathDict = new Dictionary<int, string[]>();
                                noveltyInfoDict.TryGetValue(trainingNoveltyLevel,out levelPathDict);
            /*                  if(trainingEntry.noveltyType=="*"){
                                    List<int> types = new List<int>();
                                    types.AddRange(levelPathDict.Keys);
                                    Shuffle(types);  
                                    trainingNoveltyType = types[0];
                                }
                                else{*/
                                trainingNoveltyType = int.Parse(trainingEntry.noveltyType);
            //                  }

                                string[] levelPaths;
                                List<string> noveltyBucket = new List<string>();
                                List<string> noNoveltyBucket = new List<string>();
                                List<string> noDiffBucket = new List<string>();

                                if(levelPathDict.TryGetValue(trainingNoveltyType,out levelPaths)){
                                    
                                    foreach(string level in levelPaths){
                                        
                                        string[] segs = level.Split(Path.DirectorySeparatorChar);

                                        string[] infoSegs = segs[segs.Length-1].Split('_');
                                        
                                        if(infoSegs.Length > 1){
                                            int position = int.Parse(infoSegs[1]);
                                            int diff = int.Parse(infoSegs[2]);
                                            if(position == 0){
                                                noNoveltyBucket.Add(level);
                                            }
                                            else if(diff == 0){
                                                noDiffBucket.Add(level);
                                            }
                                            else{
                                                noveltyBucket.Add(level);
                                            }
                                        }
                                        else{
                                                noveltyBucket.Add(level);
                                        }

                                    }


                                    //adjust portions
                                    if(trainingEntry.noDifferencePortion + trainingEntry.noveltyAbsencePortion >1){
                                        trainingEntry.noveltyPortion = 0;
                                        trainingEntry.noDifferencePortion = trainingEntry.noDifferencePortion/(trainingEntry.noDifferencePortion + trainingEntry.noveltyAbsencePortion);
                                        trainingEntry.noveltyAbsencePortion = trainingEntry.noveltyAbsencePortion/(trainingEntry.noDifferencePortion + trainingEntry.noveltyAbsencePortion);
                                    }
                                    else{
                                        trainingEntry.noveltyPortion = 1 - trainingEntry.noDifferencePortion - trainingEntry.noveltyAbsencePortion;
                                    }


                                    //adjust number of levels from each bucket according to the portion
                                    //especially adjust the portions if one or more buckets are empty 
                                    if(noveltyBucket.Count == 0 && noNoveltyBucket.Count == 0 && noDiffBucket.Count == 0){
                                        // do not add levels if not levels are available
                                        trainingEntry.nTrainingLevels[i] = 0;
                                    }

                                    //at lease one bucket contains one or more levels
                                    else{
                                        if(noveltyBucket.Count == 0){
                                            trainingEntry.noveltyPortion = 0;
                                            if(noNoveltyBucket.Count == 0){
                                                trainingEntry.noveltyAbsencePortion = 0;
                                                trainingEntry.noDifferencePortion = 1;
                                            }
                                            else if(noDiffBucket.Count == 0){
                                                trainingEntry.noveltyAbsencePortion = 1;
                                                trainingEntry.noDifferencePortion = 0;

                                            }
                                            else{
                                                trainingEntry.noveltyAbsencePortion = trainingEntry.noveltyAbsencePortion/(trainingEntry.noveltyAbsencePortion + trainingEntry.noDifferencePortion);
                                                trainingEntry.noDifferencePortion = trainingEntry.noDifferencePortion/(trainingEntry.noveltyAbsencePortion + trainingEntry.noDifferencePortion);
                                            }

                                        }

                                        else if(noNoveltyBucket.Count == 0){
                                            trainingEntry.noveltyAbsencePortion = 0;
                                            if(noDiffBucket.Count == 0){
                                                trainingEntry.noDifferencePortion = 0;
                                                trainingEntry.noveltyPortion = 1;
                                            }
                                            else{//no diff and novelty buckets are both not empty
                                                trainingEntry.noveltyPortion = trainingEntry.noveltyPortion/(trainingEntry.noveltyPortion+trainingEntry.noDifferencePortion);
                                                trainingEntry.noDifferencePortion = trainingEntry.noDifferencePortion/(trainingEntry.noveltyPortion+trainingEntry.noDifferencePortion);
                                            }
                                        }
                                        
                                        
                                        else if(noDiffBucket.Count == 0){
                                            //here nodiff and no novelty buckets should not be empty, as already checked before
                                            trainingEntry.noDifferencePortion = 0;
                                            trainingEntry.noveltyAbsencePortion = trainingEntry.noveltyAbsencePortion/(trainingEntry.noveltyAbsencePortion+trainingEntry.noveltyPortion);
                                            trainingEntry.noveltyPortion = trainingEntry.noveltyPortion/(trainingEntry.noveltyAbsencePortion+trainingEntry.noveltyPortion);
                                        }
                                    }
                                    
                                    int noveltyStartIndex = 0;
                                    int noNoveltyStartIndex = noveltyStartIndex + (int)Math.Round(trainingEntry.nTrainingLevels[i] * trainingEntry.noveltyPortion, MidpointRounding.AwayFromZero);  
                                    int noDiffStartIndex = noNoveltyStartIndex + (int)Math.Round(trainingEntry.nTrainingLevels[i] * trainingEntry.noveltyAbsencePortion,MidpointRounding.AwayFromZero);                             
                                    for(int k = 0; k < trainingEntry.nTrainingLevels[i]; k++){

                                        string levelPath = "";
                                        
                                        // use exactly the same path order for both notify and non notify trial pair
                                        if(trial.bothNotifyNovelty && repeatTrial < 1){
                                            //if(k > prevSelectedLevelList.Count - 1){
                                                break;
                                            //}
                                            //levelPath = prevSelectedLevelList[k];
                                        }
                                        //select from no difference bucket
                                        else if(k >= noveltyStartIndex && k < noNoveltyStartIndex && noveltyBucket.Count > 0){
                                            int levelCount = noveltyBucket.Count;
                                            int nextIndex = rnd.Next(0,levelCount);
                                            levelPath = noveltyBucket[nextIndex];
                                        }

                                        //select from novelty absence bucket
                                        else if(k >= noNoveltyStartIndex && k < noDiffStartIndex && noNoveltyBucket.Count>0){
                                            int levelCount = noNoveltyBucket.Count;
                                            int nextIndex = rnd.Next(0,levelCount);
                                            levelPath = noNoveltyBucket[nextIndex];
                                        }

                                        //select from main novelty bucket
                                        else if(k >= noDiffStartIndex && noDiffBucket.Count > 0){
                                            int levelCount = noDiffBucket.Count;
                                            int nextIndex = rnd.Next(0,levelCount);
                                            levelPath = noDiffBucket[nextIndex];
                                        }

                                        else{continue;}
                                        
                                        prevSelectedLevelList.Add(levelPath);
    //                                    currentSelectedPathList.Add(levelPath);

                                        nTotalTrainingLevels++;
                                    }         
                                    Shuffle(prevSelectedLevelList);
                                    allSelectedLevelList.AddRange(prevSelectedLevelList);
                                    prevSelectedLevelList.Clear();
                                    if(sampledDict!= null && !sampledDict.ContainsKey((trainingNoveltyLevel,trainingNoveltyType))){
                                        sampledDict.Add((trainingNoveltyLevel,trainingNoveltyType),/*indexList*/ new List<int>()/*dummy list*/);
                                    }
                                }

                            }
                        }

                        foreach (string path in allSelectedLevelList){
                                
                            writer.WriteStartElement("game_levels");
                            //string noveltyLevel = "novelty_level_"+trainingNoveltyLevel.ToString();
                            //writer.WriteAttributeString("novelty_level", noveltyLevel);
                            //string noveltyType = "type" + trainingNoveltyType.ToString();
                            //writer.WriteAttributeString("type" , noveltyType);
                            
                            writer.WriteAttributeString("level_path",path);

//                              writer.WriteAttributeString("end",indexList[k].ToString());
                            
                            //do not remove used levels and allow multiple appearence for the same level
                            //indexList.RemoveAt(0);
                            
                            writer.WriteEndElement();

                        }
                        writer.WriteFullEndElement();



                        //test level sets
        /*                 int nTimeTestLevelSets = trainingTimeLimit/checkPointTimeLimit ;
                        int nInteractionTestSets = trainingInteractionLimit/checkPointInteractionLimit ;
                        int nTestSets = nTimeTestLevelSets + nInteractionTestSets + 1;                

                        List<TestSetEntry> testEntries = new List<TestSetEntry>();
                        
                    foreach(var testEntry in testEntryArray){
                            int repeats = nTestSets*testEntry.testSetPercentage/100;
                            testEntries.AddRange(Enumerable.Repeat(testEntry, repeats));
                        }
        */

                        if(!trial.testSetOrder){
                            Shuffle(trial.testEntryArray);
                        }
                        //zero, one or multiple testing set
                        foreach(var testEntry in trial.testEntryArray){
                            writer.WriteStartElement("game_level_set");
                            writer.WriteAttributeString("mode", "test");
                            writer.WriteAttributeString("time_limit", testEntry.timeLimit.ToString());
                            writer.WriteAttributeString("total_interaction_limit",testEntry.interactionLimit.ToString());
                            writer.WriteAttributeString("attempt_limit_per_level", testEntry.attemptPerLevel.ToString());
                            writer.WriteAttributeString("allow_level_selection",testEntry.allowLevelSelection.ToString());

                            int testNoveltyLevel;
        /*                    if(testEntry.noveltyLevel=="*"){
                                List<int> levels = new List<int>();
                                levels.AddRange(noveltyInfoDict.Keys);
                                Shuffle(levels);  
                                testNoveltyLevel = levels[0];
                            }
                            else if(testEntry.noveltyLevel=="trained"){
                                testNoveltyLevel = trainingNoveltyLevel;
                            }
                            else{*/
                                testNoveltyLevel = int.Parse(testEntry.noveltyLevel);
                            //}

                            int testNoveltyType;
                            Dictionary<int,string[]> testLevelPathDict = new Dictionary<int, string[]>();
                            noveltyInfoDict.TryGetValue(testNoveltyLevel,out testLevelPathDict);
        /*                  if(testEntry.noveltyType == "*"){
                                List<int> types = new List<int>();
                                types.AddRange(testLevelPathDict.Keys);
                                Shuffle(types);  
                                testNoveltyType = types[0];
                            }
                            else if(testEntry.noveltyType=="trained"){
                                testNoveltyType = trainingNoveltyType;
                            }
                            else{*/
                                testNoveltyType = int.Parse(testEntry.noveltyType);
                            //}

                            //write get levels elements
                            string[] testLevelPaths; 
                            if(testLevelPathDict.TryGetValue(testNoveltyType,out testLevelPaths)){
                                var testIndexList = new List<int>();

                                if(!sampledDict.TryGetValue((testNoveltyLevel,testNoveltyType),out testIndexList)){
                                    testIndexList = new List<int>();
                                    for(int k = 0; k < testLevelPaths.Length; k++){
                                        testIndexList.Add(k);
                                    }
                                    Shuffle(testIndexList);
                                    sampledDict.Add((testNoveltyLevel,testNoveltyType),testIndexList);
                                }


                                if(testEntry.nLevels > testIndexList.Count){
                                    testEntry.nLevels = testIndexList.Count;
                                }

                                for(int k = 0; k < testEntry.nLevels; k++){
                                    writer.WriteStartElement("game_levels");
                                    string noveltyLevel = "novelty_level_"+testNoveltyLevel.ToString();
                                    writer.WriteAttributeString("novelty_level", noveltyLevel);
                                    string noveltyType = "type" + testNoveltyType.ToString();
                                    writer.WriteAttributeString("type" , noveltyType);
                                    writer.WriteAttributeString("start",testIndexList[0].ToString());
                                    writer.WriteAttributeString("end",testIndexList[0].ToString());
                                    testIndexList.RemoveAt(0);
                                    writer.WriteEndElement();
                                    
                                }                

                            }
                            writer.WriteFullEndElement();
                        }
                        //</trial>
                        writer.WriteFullEndElement();
                        if(repeatTrial > 0){
                            trial.notifyNovelty = true;//for both notify and not notify
                        }

                    }    
                }
                
            
            }

		}
		
		StreamWriter streamWriter;
        streamWriter = new StreamWriter(configPath);
		streamWriter.WriteLine(output.ToString());
		streamWriter.Close();

    }

    private void readTestGenerationConfig(string configMetaPath){
//        writeMetaConfigFile("./configMeta.xml");
        
        string configPath =  ABConstants.STREAMINGASSETS_FOLDER + "/config.xml";
        string configXml = File.ReadAllText(configMetaPath);
        XmlReaderSettings readerSettings = new XmlReaderSettings();
        readerSettings.IgnoreComments = true;
//        readerSettings.IgnoreProcessingInstructions = true;
        readerSettings.IgnoreWhitespace = true;
        XmlReader reader = XmlReader.Create(new StringReader(configXml),readerSettings);
        reader.ReadToFollowing("config_generation");
        
        if(reader.GetAttribute("repeats")!=null){
            nEvaluations = int.Parse(reader.GetAttribute("repeats"));
        }
        else{
            nEvaluations = 1;
        }

        reader.ReadToFollowing("novelty_detection_measurement");
        if(reader.GetAttribute("step")!=null){
            reader.MoveToAttribute("step");
            noveltyDetectionReportStep = int.Parse(reader.Value);
        }
        else{
            noveltyDetectionReportStep = 0;
        }

        if(reader.GetAttribute("measure_in_training")!=null){
            reader.MoveToAttribute("measure_in_training");
            string value = reader.Value;

            if(value == "true"){
                trainingMeasureNoveltyLikelihood = true;
            }
            else{
                trainingMeasureNoveltyLikelihood = false;
            }
        }
        else{
            trainingMeasureNoveltyLikelihood = false;
        }
        if(reader.GetAttribute("measure_in_testing")!=null){
            reader.MoveToAttribute("measure_in_testing");
            string value = reader.Value;
            if(value == "true"){
                testMeasureNoveltyLikelihood = true;
            }
            else{
                testMeasureNoveltyLikelihood = false;
            }
        }
        else{
            testMeasureNoveltyLikelihood = false;
        }
        
        reader.ReadToFollowing("trials");
        while(!reader.EOF){
            //read to trial element
            reader.Read();
            string nodeName = reader.LocalName;
            if(nodeName =="trials"){
                break;
            }
            Trial trial = new Trial(); 
            if(reader.GetAttribute("notify_novelty")!=null){
                reader.MoveToAttribute("notify_novelty");
                string value = reader.Value;
                if(value == "true"){
                    trial.notifyNovelty = true;
                }
                else if (value == "both"){
                    trial.bothNotifyNovelty = true;
                }

                else {
                    trial.notifyNovelty = false;
                }
            }



            if(reader.ReadToFollowing("count")){
                trial.trialCount = reader.ReadElementContentAsInt();
            }
            else{
                trial.trialCount = 1;
            }
            trial.overallTrainingTimeLimit = new int[trial.trialCount];

    //        if(reader.ReadToFollowing("repeats")){
                //the reader has moved to repeats element!
                trial.trialRepeats = reader.ReadElementContentAsInt();
    //        }

    //        if(reader.ReadToFollowing("checkpoint")){
                //the reader is at checkpoint element
                reader.Read();
                while(!reader.EOF){
                    nodeName = reader.LocalName;
                    if(nodeName == "checkpoint"){
                        break;
                    }
                    
                    string type = reader.GetAttribute("type");
                    int value = reader.ReadElementContentAsInt();
                    if(type == "time"){
                        trial.checkPointTimeLimit = value;
                    }
                    else if(type == "interaction")
                    {
                        trial.checkPointInteractionLimit = value;                    
                    }
                }
    //        }

            if(reader.ReadToFollowing("training")){
                if(reader.ReadToFollowing("limits")){
                    reader.Read();
                    //read limit
                    while(!reader.EOF){
                        nodeName = reader.LocalName;
                        if(nodeName =="limits"){
                            break;
                        }
                        string type = reader.GetAttribute("type");
                        string value = reader.ReadElementContentAsString();
                        if(type == "time"){
                            trial.avgTrainingTimeLimitPerLevel = int.Parse(value);
                        }
                        else if(type == "interaction")
                        {
                            trial.trainingInteractionLimit = int.Parse(value);                    
                        }
                        else if(type=="attempt_per_level"){
                            trial.trainingAttemptPerLevel = int.Parse(value);
                        }
                        else if(type=="allow_level_selection"){
                            if(value == "true"){
                                trial.trainingAllowLevelSelection = true;
                                
                            }
                            else{
                                trial.trainingAllowLevelSelection = false;
                            }
                        }
                    }
                    
                }
                if(reader.ReadToFollowing("game_level_sets")){
 
                    while(reader.Read()){
                        nodeName = reader.LocalName;
                        if(nodeName =="game_level_sets"){
                            break;
                        }
                        TrainingEntry trainingEntry = new TrainingEntry();
                        trainingEntry.allowLevelSelection = trial.trainingAllowLevelSelection;
                        trainingEntry.attemptPerLevel = trial.trainingAttemptPerLevel;
                        trainingEntry.interactionLimit = trial.trainingInteractionLimit;
                        
                        if(reader.GetAttribute("novelty_level")!=null){
                            string value = reader.GetAttribute("novelty_level");
                            trainingEntry.noveltyLevel = value;
                        }
                        if(reader.GetAttribute("novelty_type")!=null){
                            string value = reader.GetAttribute("novelty_type");
                            trainingEntry.noveltyType = value;
                        
                        }
                        if(reader.GetAttribute("amount")!=null){

                            string value =  reader.GetAttribute("amount");
                            string[] range = value.Split('-');
                            int amount = 0;

                            trainingEntry.nTrainingLevels = new int[trial.trialCount];
                            trainingEntry.timeLimit = new int[trial.trialCount];
                            for(int c = 0; c < trial.trialCount; c++){
                                if(range.Length > 1){
                                    int low = int.Parse(range[0]);
                                    int high = int.Parse(range[1]);
                                    amount = rnd.Next(low,high);
                                }
                                else{
                                    amount = int.Parse(value); 
                                }
                                
                                trainingEntry.nTrainingLevels[c] = amount;
                                trainingEntry.timeLimit[c] = trial.avgTrainingTimeLimitPerLevel * amount;
                            }
                        }
                        if(reader.GetAttribute("novelty_absence_portion")!=null){
                            reader.MoveToAttribute("novelty_absence_portion");
                            trainingEntry.noveltyAbsencePortion = float.Parse(reader.Value);
                        }

                        if(reader.GetAttribute("no_diff_portion")!=null){
                            reader.MoveToAttribute("no_diff_portion");
                            trainingEntry.noDifferencePortion = float.Parse(reader.Value);
                        }
                        for(int c = 0; c < trial.trialCount; c++){
                            trial.overallTrainingTimeLimit[c] += trainingEntry.timeLimit[c];
                        }
                        trial.trainingEntryArray.Add(trainingEntry);
                    }
                }
                
            }

            if(reader.ReadToFollowing("test")){
                if(reader.GetAttribute("ordered")!=null){
                    string value = reader.GetAttribute("ordered");
                    if(value == "true"){
                        trial.testSetOrder = true;
                    }
                    else {
                        trial.testSetOrder = false;
                    }
                }

                if(reader.ReadToFollowing("game_level_sets")){
                    while(reader.Read()){
                        nodeName = reader.LocalName;
                        if(nodeName =="game_level_sets"){
                            break;
                        }
                        TestSetEntry testEntry = new TestSetEntry();

                        if(reader.GetAttribute("novelty_level")!=null){
                            string value = reader.GetAttribute("novelty_level");
                            testEntry.noveltyLevel = value;

                        }
                        if(reader.GetAttribute("novelty_type")!=null){
                            string value = reader.GetAttribute("novelty_type");
                            testEntry.noveltyType = value;                
                        }
                
                        if(reader.GetAttribute("amount")!=null){
                            string value =  reader.GetAttribute("amount");
                            string[] range = value.Split('-');
                            int amount = 0;
                            if(range.Length > 1){
                                int low = int.Parse(range[0]);
                                int high = int.Parse(range[1]);
                                amount = rnd.Next(low,high);
                            }
                            else{
                                amount = int.Parse(value); 
                            }
                            testEntry.nLevels = amount;
                        }
                        if(reader.GetAttribute("set_appearance_percentage")!=null){
                            int value = int.Parse(reader.GetAttribute("set_appearance_percentage"));
                            testEntry.testSetPercentage = value;
                        }
                        if(reader.GetAttribute("time")!=null){
                            int value = int.Parse(reader.GetAttribute("time"));
                            testEntry.timeLimit = value;                
                        }
                        if(reader.GetAttribute("interaction")!=null){
                            int value = int.Parse(reader.GetAttribute("interaction"));
                            testEntry.interactionLimit = value;                
                        }
                        if(reader.GetAttribute("attempt_per_level")!=null){
                            int value = int.Parse(reader.GetAttribute("attempt_per_level"));
                            testEntry.attemptPerLevel = value;                
                        }
                        if(reader.GetAttribute("allow_level_selection")!=null){
                            string value = reader.GetAttribute("allow_level_selection");
                            if(value=="true"){
                                testEntry.allowLevelSelection = true;                
                            }
                            else{
                                testEntry.allowLevelSelection = false;
                            }
                        }
                        trial.testEntryArray.Add(testEntry);
                    }
                }
            }

            //read over </test> and </trial>
            reader.Read();
            reader.Read();
            trials.Add(trial);
            writeConfigFile(configPath);

        }    
    }
    
}

