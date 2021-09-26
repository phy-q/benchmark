using UnityEngine;
using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;


public static class ABLevelUpdate{
    public static List<string> levelPath = new List<string>();
    public static string[] allXmlFiles = null; 
    private static DateTime lastModifiedTimeResource;
    private static DateTime lastModifiedTimeStreaming;

    public static int RefreshLevelList()
    {
        allXmlFiles = getAllXmlFiles();
        LevelList.Instance.RefreshLevelsFromSource(allXmlFiles);
        return allXmlFiles.Length;

    }

    public static string[] getAllXmlFiles(){
        DirectoryInfo infoResource = new DirectoryInfo(ABConstants.DEFAULT_LEVELS_FOLDER);
        DirectoryInfo infoStreaming = new DirectoryInfo(Application.dataPath + ABConstants.CUSTOM_LEVELS_FOLDER);

        DateTime thisModifiedTimeResource = infoResource.LastWriteTime;
        DateTime thisModifiedTimeStreaming = infoStreaming.LastWriteTime;
        
        // Debug.Log("last resource " + lastModifiedTimeResource);
        // Debug.Log("this resource " + thisModifiedTimeResource);
        // Debug.Log("last stream " + lastModifiedTimeStreaming);
        // Debug.Log("this stream " + thisModifiedTimeStreaming);

        if(thisModifiedTimeResource!=lastModifiedTimeResource || 
            thisModifiedTimeStreaming!=lastModifiedTimeStreaming && allXmlFiles!=null){

                Debug.Log("file(s) in Levels folder have been changed, rescanning levels... ");
                lastModifiedTimeStreaming = thisModifiedTimeStreaming;
                lastModifiedTimeResource = thisModifiedTimeResource;


                Debug.Log("Streamingassets modified time " + thisModifiedTimeStreaming);
                levelPath.Clear();
                List<string> resourcesXml = getLevelXMLList(ABConstants.DEFAULT_LEVELS_FOLDER);
                List<string> streamingXml = new List<string>();
#if UNITY_WEBGL && !UNITY_EDITOR

		    // WebGL builds does not load local files

#else
            // Load levels in the streaming folder
            streamingXml = getLevelXMLList(Application.dataPath + ABConstants.CUSTOM_LEVELS_FOLDER);

#endif

            // Combine the two sources of levels
            List<string> allXmlFilesList = new List<string>(resourcesXml.Count + streamingXml.Count);
            allXmlFilesList.AddRange(resourcesXml);
            allXmlFilesList.AddRange(streamingXml);
            lastModifiedTimeResource = infoResource.LastWriteTime; 
            lastModifiedTimeStreaming = infoStreaming.LastWriteTime;
            allXmlFiles = allXmlFilesList.ToArray();
        }
        return allXmlFiles;
    }

    private static List<string> getLevelXMLList(string parentPath){
		
        List<string> xmlStrings= new List<string>();
        
        if(!Directory.Exists(parentPath)){
            return xmlStrings;
        }


        //sorted by name
        // string[] noveltyLevelDirs = Directory.GetDirectories(parentPath, "novelty_level_*", SearchOption.TopDirectoryOnly).OrderBy(f => Regex.Replace(f, @"\d+", n => n.Value.PadLeft(4, '0'))).ToArray();

        // for fast physical test, only use the folder 0.
        string[] noveltyLevelDirs = Directory.GetDirectories(parentPath, "novelty_level_0*", SearchOption.TopDirectoryOnly).OrderBy(f => Regex.Replace(f, @"\d+", n => n.Value.PadLeft(4, '0'))).ToArray();

        foreach (string levelDir in noveltyLevelDirs){
			//sorted by name
            string[] noveltyTypeDirs = Directory.GetDirectories(levelDir, "type*", SearchOption.TopDirectoryOnly).OrderBy(f => Regex.Replace(f, @"\d+", n => n.Value.PadLeft(4, '0'))).ToArray();


//            Debug.Log(levelDir);
            foreach (string typeDir in noveltyTypeDirs){
				// Load levels in the resources folder

                string assetBundleFilePath = typeDir+ABConstants.assetBundleOSPath;
                string[] levelFiles = Directory.GetFiles(typeDir+"/Levels/", "*.xml");
                levelPath.AddRange(levelFiles);
//                Debug.Log(typeDir);
//                Debug.Log("n levels " + levelFiles.Length);
                for (int i = 0; i < levelFiles.Length; i++){
                    string data = File.ReadAllText(levelFiles[i]);
                    String[] spearator = {"<GameObjects>"};
                    string[] dataArray = data.Split(spearator,2,StringSplitOptions.RemoveEmptyEntries); 
                    data= dataArray[0] + "<assetBundle path=\"" +assetBundleFilePath+"\" />\n<GameObjects>\n" + dataArray[1];
					//Debug.Log("data " + data);
                    xmlStrings.Add(data);
                }
			}	
		}
		
		return xmlStrings;
	}

}
