using System.Collections.Generic;
using UnityEngine.SceneManagement;

public class TrialInfo
{
    public int id;
    public List<TestSet> testLevelSets;
    public int currentTestSetIndex;
    public TrainingSet trainingSet;
    public int currentExecutionIndex;
    public bool isCheckPoint;
    public int nExecutions;

    //if the agent has tried to load level when there is no available level.
    public bool loadLevelAfterFinished;

//    public bool isTraining;

    public bool hasCheckpointTimeLimit;
    public int checkpointTimeLimit;
    public bool hasCheckpointInteractionLimit;
    public int checkpointInteractionLimit;
    public bool hasCheckpointLevelReachedLimit;
    public int checkpointLevelReachedLimit;
    public bool notifyNovelty;

    public TrialInfo()
    {
        testLevelSets = new List<TestSet>();
        trainingSet = null;
        currentExecutionIndex = 1;
        currentTestSetIndex = 0;
        isCheckPoint = false;
        notifyNovelty = false; //do not tell the agent info about novelty by default  
        loadLevelAfterFinished = false;
    }

    public void reset(){
//        currentExecutionIndex = 1; do not reset this one
        currentTestSetIndex = 0;
        isCheckPoint = false;
        if(trainingSet!= null){
            trainingSet.resetAvailableLevelList();
        }
        if(testLevelSets!=null){
            foreach(var testLevelSet in testLevelSets){
                if(testLevelSet!=null){
                    testLevelSet.resetAvailableLevelList();
                }
            }
        }
    }

    //if all levels in a training set has been played
    public bool isTrainingFinished(){
        bool completed = false;

        string currentScence = SceneManager.GetActiveScene().name;

        if(loadLevelAfterFinished){
            completed = true;
            loadLevelAfterFinished = false;
        }

        else if(currentScence=="GameWorld"){
            if(trainingSet.availableLevelList.Count<=0 &&
            (ABGameWorld.Instance.LevelCleared()||ABGameWorld.Instance.LevelFailed())){
                completed = true;
            }
        } 
        

        return completed;
    }

    //if all levels in a test set has been played
    public bool isTestSetFinished(){
        bool completed = false;

        if(loadLevelAfterFinished){
            completed = true;
            loadLevelAfterFinished = false;
        }

        else if (testLevelSets[currentTestSetIndex].availableLevelList.Count<=0 &&
        (ABGameWorld.Instance.LevelCleared()||ABGameWorld.Instance.LevelFailed())){
            completed = true;
        }

        return completed;
    }

    public void update(){
        
    }

    public bool isCheckPointReached(){
        
        
        return isCheckPoint;
    }
}