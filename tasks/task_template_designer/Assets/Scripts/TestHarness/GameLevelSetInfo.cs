using System.Collections.Generic;
using System.IO;
using System;
using System.Diagnostics;
public class GameLevelSetInfo
{
    public string[] levelSetPathArray{set;get;}

    public Mode mode{set;get;}

    public int currentLevelIndex;

    public bool hasTimeLimit;
    public int timeLimit{set;get;}
    public int timeElapsed;
    public bool hasInteractionLimit;
    public int interactionLimit{set;get;}
    public int interactionUsed;
    public bool notifyNovelty;
    public bool mustSolve;//must solve the level to access the next, only used when the levels are as a sequence
    
    public bool hasLevelAttepmtLimit;
    public int levelAttemptLimit;
    public int totalLevelAttempts;
    public bool hasLevelTimeLimit;
    public int levelTimeLimit;

    public bool isSequence{set;get;}

    public List<string> levelPaths{set;get;}
    public LinkedList<int> availableLevelList;//start from 1
    public int[] nLevelEntryList;//record how many times the level is tried
    public int[] levelAttempts;
    public int[] levelTimeUsed;


    public enum Mode { TRAINING, TEST, UNKNOWN};
    public GameLevelSetInfo()
    {
        resetAvailableLevelList();
    }

    public List<string> getLevelSetXmlData(){
        List<string> allXmlData = new List<string>();
        string spliter = "Levels";
        for(int i = 0; i < levelPaths.Count; i++){
            int lastIndex = levelPaths[i].LastIndexOf(spliter) - 1;
            string assetBundlePath = levelPaths[i].Substring(0,lastIndex) + ABConstants.assetBundleOSPath;

            string data = File.ReadAllText(levelPaths[i]);
            string[] spearator = {"<GameObjects>"};
            string[] dataArray = data.Split(spearator,2,StringSplitOptions.RemoveEmptyEntries); 
            data= dataArray[0] + "<assetBundle path=\"" +assetBundlePath+"\" />\n<GameObjects>\n" + dataArray[1];
            //UnityEngine.Debug.Log("data " + data);
            allXmlData.Add(data);
        }
        
        return allXmlData;
    }

    public void updateAvailableLevelList(){

    }

    public bool resetAvailableLevelList(){
        
        if(levelPaths==null) return false;
        
        nLevelEntryList = new int[levelPaths.Count];
        availableLevelList = new LinkedList<int>();
        totalLevelAttempts = levelAttemptLimit*levelPaths.Count;
        for(int i = 0; i < levelPaths.Count; i++){
            availableLevelList.AddLast(i);
            nLevelEntryList[i] = this.levelAttemptLimit;
        }
        return true;

    }

}