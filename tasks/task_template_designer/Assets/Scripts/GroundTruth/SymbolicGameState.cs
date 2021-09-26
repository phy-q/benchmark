/**
* @author Peng Zhang
*
* @date - 04/08/2020 
*/

using System.Collections.Generic;
using System;
using UnityEngine;
using System.IO;
using System.Linq;

public class SymbolicGameState
{
    Camera cam;
    private bool devMode;
    private bool useNoise;

    public ColorData colorData;

    private enum ObjectType { Pig, Bird, SlingShot, Platform, Ground, Block};
    public SymbolicGameState(bool noise = true){
        this.cam = Camera.main.gameObject.GetComponent<Camera>();
        string colorDataFile = ABConstants.colorDataFile;
        string colorHistograms = File.ReadAllText(colorDataFile);
        this.colorData = JsonUtility.FromJson<ColorData>(colorHistograms);
        this.colorData.readColorMaps();
        this.devMode = LoadLevelSchema.Instance.devMode;
        this.useNoise = noise;
    }

    public string GetGTJson(bool useNoise = true){
        string gtJson = "[{\"type\": \"FeatureCollection\",\"features\": [";
        
        //Get ground
        if (GameObject.Find("Ground") != null && cam != null)
        {
            GameObject ground = GameObject.Find("Ground");
            if (ground.GetComponent<BoxCollider2D>() != null)
            {
                string groundID = ground.GetInstanceID().ToString();

                var Screenmax = cam.WorldToScreenPoint(ground.GetComponent<BoxCollider2D>().bounds.max);
                int yind = (int)Mathf.Max(Mathf.Round(Screenmax.y), 0.0f);
                
                yind = Screen.height - yind;
                GTGround gtGround = new GTGround(groundID, "Ground",yind);
                
                gtJson = gtJson + gtGround.ToJsonString(this.devMode) + ",";
            }
        }

        //Get sling shot
        GameObject[] Slingshot = GameObject.FindGameObjectsWithTag("Slingshot");
        float xmax=0,xmin=0,ymax=0,ymin=0;
        foreach (GameObject gameObject in Slingshot)
        {
            if (gameObject.name == "slingshot_back")
            {
                Vector3 objectBoundmin = cam.WorldToScreenPoint(gameObject.GetComponent<Renderer>().bounds.min);
                Vector3 objectBoundmax = cam.WorldToScreenPoint(gameObject.GetComponent<Renderer>().bounds.max);

                xmax = Mathf.Round(objectBoundmax.x);
                //if (xmax < 0) { xmax = 0; }
                ymin = Mathf.Round(objectBoundmin.y);
                //if (ymin < 0) { ymin = 0; }
            }
            else if (gameObject.name == "slingshot_front")
            {
                Vector3 objectBoundmin = cam.WorldToScreenPoint(gameObject.GetComponent<Renderer>().bounds.min);
                Vector3 objectBoundmax = cam.WorldToScreenPoint(gameObject.GetComponent<Renderer>().bounds.max);

                ymax = Mathf.Round(objectBoundmax.y);
                //if (ymax < 0) { ymax = 0; }
                xmin = Mathf.Round(objectBoundmin.x);
                //if (xmin < 0) { xmin = 0; }
            }
        }
        Vector2[] slingshotBound = new Vector2[4];
        slingshotBound[0] = new Vector2(xmin, Screen.height - ymin);
        slingshotBound[1] = new Vector2(xmin, Screen.height - ymax);
        slingshotBound[2] = new Vector2(xmax, Screen.height - ymax);
        slingshotBound[3] = new Vector2(xmax, Screen.height - ymin);

        List<Vector2[]> slingPaths = new List<Vector2[]>();
        slingPaths.Add(slingshotBound);
        string slingID = Slingshot[0].GetInstanceID().ToString();
        //colorMaps[] SlingShotColors = ColorPoints(vert,false);
        float slingHP = float.MaxValue;
        GTObject gtSling = new GTObject(slingID,"Slingshot",null,slingPaths,slingHP);
        gtJson = gtJson + gtSling.ToJsonString(this.devMode) + ",";

        //Get trajectory points
        GameObject[] Trajectory = GameObject.FindGameObjectsWithTag("Trajectory");
        string trajID = "";
        Vector2[] screenLocation = new Vector2[Trajectory.Length];

        for (int i=0; i<Trajectory.Length; i++)
        {
            GameObject gameObject = Trajectory[i];
            if(!gameObject.GetComponent<Renderer>().isVisible){
                continue;
            }

            if(i==0){
                trajID = gameObject.GetInstanceID().ToString();
            }
            var screenPosition = cam.WorldToScreenPoint(gameObject.transform.position);
            if (useNoise)
            {
                screenLocation[i] = new Vector2(screenPosition.x, Screen.height - screenPosition.y) + UnityEngine.Random.insideUnitCircle * 2;
            }
            else
            {
                screenLocation[i] = new Vector2(screenPosition.x, Screen.height - screenPosition.y);
            }
        }
        //only include trajectory when there is at lease one trajectory point detected
        if(Trajectory.Length > 0){
            GTTrajectory gtTrajectory = new GTTrajectory(trajID,"Trajectory", screenLocation);
            gtJson = gtJson + gtTrajectory.ToJsonString(this.devMode)+",";
        }

        //Get pigs
        GameObject[] pigsSmall = GameObject.FindGameObjectsWithTag("PigSmall");
        GameObject[] pigBig = GameObject.FindGameObjectsWithTag("PigBig");
        GameObject[] pigMedium = GameObject.FindGameObjectsWithTag("PigMedium");

        GameObject[] pigs = pigsSmall.Concat(pigBig).Concat(pigMedium).ToArray();
        foreach (GameObject gameObject in pigs)
        {   
            bool outOfBound = false;
            string blockName = gameObject.GetComponent<SpriteRenderer>().sprite.name;
            if (gameObject.name == "BasicSmall(Clone)" || gameObject.name == "BasicMedium(Clone)" || gameObject.name == "BasicBig(Clone)")
            {
                float hp = gameObject.GetComponent<ABGameObject>().getCurrentLife();
                string pigID = gameObject.GetInstanceID().ToString();
                int pathCount = gameObject.GetComponent<PolygonCollider2D>().pathCount;
                List<Vector2[]> paths = new List<Vector2[]>();
//                ObjectContour[] noisyPaths = new ObjectContour[pathCount];
                for(int i = 0; i < pathCount; i++ ){
                    Vector2[] objPoints = gameObject.GetComponent<PolygonCollider2D>().GetPath(i);
                    
                    //vertices that is in unity coordinate system
                    //the origin is buttom-left corner 
                    Vector3[] screenPoints = new Vector3[objPoints.Length];
                    
                    Vector2[] noisePoints = new Vector2[objPoints.Length];

                    for (int j = 0; j < objPoints.Length; j++)
                    {
                        screenPoints[j] = cam.WorldToScreenPoint(gameObject.transform.TransformPoint(objPoints[j]));
                        objPoints[j] = new Vector2(Mathf.Round(screenPoints[j].x), Mathf.Round(Screen.height - screenPoints[j].y));                        
                    }

                    //skip this object if it is out of the screen
                    if(IsOutOfBound(objPoints)){
                        outOfBound = true;
                        break;
                    }

                    if(this.useNoise){
                        objPoints = ApplyNoise(objPoints);                    
                    }                       
                    paths.Add(objPoints);

                }
                if(outOfBound){
                    continue;
                }
                //var objColors = ColorPoints(gameObject, objPoints, false);
                ColorEntry[] objColors = ColorDataLookUp(blockName,this.useNoise);

                string objectType = "Object";
                if(this.devMode){
                    objectType = blockName;
                }                    
                GTObject pig = new GTObject(pigID, objectType, objColors, paths, hp);
                gtJson = gtJson + pig.ToJsonString(this.devMode) + ',';
            }
        }

        //Get birds
        GameObject[] birds = GameObject.FindGameObjectsWithTag("Bird");
        foreach (GameObject gameObject in birds)
        {
            string blockName = gameObject.GetComponent<SpriteRenderer>().sprite.name;
            //ad-hoc code for distinguishing the novel bird that uses pig sprite  
            if(blockName=="pig_basic_small_2"){
                blockName = "novel_object_0_1_9";
            }

            if (gameObject.GetComponent<SpriteRenderer>().color.a != 0)
            {   
                bool outOfBound = false;
                float hp = gameObject.GetComponent<ABGameObject>().getCurrentLife();
                //filter out invisible objects
                // if(!gameObject.GetComponent<Renderer>().isVisible){
                //     continue;
                // }
                string birdID = gameObject.GetInstanceID().ToString();
                int pathCount = gameObject.GetComponent<PolygonCollider2D>().pathCount;
                List<Vector2[]> paths = new List<Vector2[]>();
                for(int i = 0; i < pathCount; i++ ){
                    Vector2[] objPoints = gameObject.GetComponent<PolygonCollider2D>().GetPath(i);
                    Vector3[] screenPoints = new Vector3[objPoints.Length];
                    Vector2[] noisePoints = new Vector2[objPoints.Length];

                    for (int j = 0; j < objPoints.Length; j++)
                    {
                        screenPoints[j] = cam.WorldToScreenPoint(gameObject.transform.TransformPoint(objPoints[j]));
                        objPoints[j] = new Vector2(Mathf.Round(screenPoints[j].x), Mathf.Round(Screen.height - screenPoints[j].y));
                    }
                    //skip this object if it is out of the screen
                    if(IsOutOfBound(objPoints)){
                        outOfBound = true;
                        break;
                    }
                    paths.Add(objPoints);
                }

                if(outOfBound){
                    continue;
                }
                ColorEntry[] objColors = ColorDataLookUp(blockName,false);
                
                string objectType = "Object";
                if(this.devMode){
                    objectType = blockName;
                }
                
                GTObject bird = new GTObject(birdID,objectType,objColors,paths, hp);
                gtJson = gtJson + bird.ToJsonString(this.devMode) + ',';
            }

        }

        //Get blocks
        GameObject[] cricleBlocks = GameObject.FindGameObjectsWithTag("Circle");
        GameObject[] triangleBlocks = GameObject.FindGameObjectsWithTag("Triangle");
        GameObject[] haloSquareBlocks = GameObject.FindGameObjectsWithTag("SquareHole");
        GameObject[] novelBlocks = GameObject.FindGameObjectsWithTag("Block");
        GameObject[] blocks = cricleBlocks.Concat(triangleBlocks).Concat(haloSquareBlocks).Concat(novelBlocks).ToArray();
        foreach (GameObject gameObject in blocks)
        {   
            bool outOfBound = false;
            if(gameObject.tag!="Circle"&&gameObject.tag!="Triangle"&&gameObject.tag!="SquareHole"&&gameObject.tag!="Block"){continue;}
            if (gameObject.name.Equals("BasicSmall(Clone)")){continue;} //pigs are removed 
            string blockName = gameObject.GetComponent<SpriteRenderer>().sprite.name;
            //adjust the name of the novel objects
            if(gameObject.tag == "Block"){
                blockName = gameObject.name;
                blockName = blockName.Replace("(Clone)","").Trim();
                ABLevel currentLevel = LevelList.Instance.GetCurrentLevel();
                string assetBundlePath = currentLevel.assetBundleFilePath;
                
                string[] strArray = assetBundlePath.Split(new string[] { "type" }, StringSplitOptions.None);
                
                string noveltyTypeStr = strArray[strArray.Length-1].Split(new string[] { "AssetBundle" }, StringSplitOptions.None)[0];
                noveltyTypeStr = noveltyTypeStr.Substring(0, noveltyTypeStr.Length - 1);//remove the file separator

                string noveltyLevelStr = strArray[strArray.Length-2].Split(new string[] { "novelty_level_" }, StringSplitOptions.None).Last();
                noveltyLevelStr = noveltyLevelStr.Substring(0, noveltyLevelStr.Length - 1);//remove the file separator

                blockName = blockName + "_" + noveltyLevelStr + "_" + noveltyTypeStr;
            }
            string blockID = gameObject.GetInstanceID().ToString();
            float hp = gameObject.GetComponent<ABGameObject>().getCurrentLife();
            int pathCount = gameObject.GetComponent<PolygonCollider2D>().pathCount;
            List<Vector2[]> paths = new List<Vector2[]>();
            for(int i = 0; i < pathCount; i++ ){
                Vector2[] objPoints = gameObject.GetComponent<PolygonCollider2D>().GetPath(i);
                Vector3[] screenPoints = new Vector3[objPoints.Length];
                Vector2[] noisePoints = new Vector2[objPoints.Length];
                
                for (int j = 0; j < objPoints.Length; j++)
                {
                    screenPoints[j] = cam.WorldToScreenPoint(gameObject.transform.TransformPoint(objPoints[j]));
                    objPoints[j] = new Vector2(Mathf.Round(screenPoints[j].x), Mathf.Round(Screen.height - screenPoints[j].y));                        
                }
                //skip this object if it is out of the screen
                if(IsOutOfBound(objPoints)){
                    outOfBound = true;
                    break;
                }                
                if(this.useNoise){
                    objPoints = ApplyNoise(objPoints);
                }
                paths.Add(objPoints);

            }
            if(outOfBound){
                continue;
            }
            ColorEntry[] objColors = ColorDataLookUp(blockName,this.useNoise);

            string objectType = "Object";
            if(this.devMode){
                objectType = blockName;
            }

            GTObject block = new GTObject(blockID, objectType, objColors, paths, hp);
            gtJson = gtJson + block.ToJsonString(this.devMode) + ',';

        }
        GameObject[] rectangleBlocks = GameObject.FindGameObjectsWithTag("Rect");
        GameObject[] TNTBlocks = GameObject.FindGameObjectsWithTag("TNT");
        GameObject[] Platforms = GameObject.FindGameObjectsWithTag("Platform");
        GameObject[] allRectBlocks = rectangleBlocks.Concat(TNTBlocks).Concat(Platforms).ToArray();
        foreach (GameObject gameObject in allRectBlocks)
        {
            string blockName = gameObject.GetComponent<SpriteRenderer>().sprite.name;
            List<Vector2[]> paths = new List<Vector2[]>();
            Vector3[] v = new Vector3[4];
            Vector3[] screenPoints = new Vector3[4];
            Vector2[] objPoints = new Vector2[4];
            if (gameObject.GetComponent<RectTransform>() == null)
            {
                gameObject.AddComponent<RectTransform>();
            }
            gameObject.GetComponent<RectTransform>().GetWorldCorners(v);
            for (int i = 0; i < 4; i++)
            {
                screenPoints[i] = cam.WorldToScreenPoint(v[i]);
                objPoints[i] = new Vector2(Mathf.Round(screenPoints[i].x), Mathf.Round(Screen.height - screenPoints[i].y));
            }
            //skip this object if it is out of the screen
            if(IsOutOfBound(objPoints)){
                continue;
            }
            if(this.useNoise){
                objPoints = ApplyNoise(objPoints);
            }
            paths.Add(objPoints);
            string blockID = gameObject.GetInstanceID().ToString();
            ABGameObject abGameObject = gameObject.GetComponent<ABGameObject>();
            float hp;
            if(abGameObject!=null){
                hp = abGameObject.getCurrentLife();
            } 
            else{//return hp as MaxValue if the object is unbreakable  
                hp = float.MaxValue;
            }

            ColorEntry[] objColors = ColorDataLookUp(blockName,this.useNoise);

            string objectType = "Object";
            if(this.devMode){
                if(blockName.Equals("effects_21")){
                    objectType = "Platform";                        
                }

                else if (blockName.Equals("effects_34")){
                    objectType = "TNT";
                }
                else{
                    objectType = blockName;
                }
            }                   
            GTObject rectBlock = new GTObject(blockID, objectType, objColors, paths, hp);
            gtJson = gtJson + rectBlock.ToJsonString(this.devMode)+ ',';
        }
        //remove last ,
        gtJson = gtJson.Remove(gtJson.Length-1,1);

        gtJson += "]}]";//end features 
        gtJson = RepresentationNovelty.addNoveltyToJSONGroundTruthIfNeeded(gtJson);
        return gtJson;
    }


    //Apply noise for a list of points
    public Vector2[] ApplyNoise(Vector2[] coords)
    {
        Vector2[] points = new Vector2[coords.Length];

        Vector2 noiseShift = UnityEngine.Random.insideUnitCircle * 2;
        for (int i = 0; i < coords.Length; i++)
        {
            points[i] = new Vector2(Mathf.Round(coords[i].x + noiseShift.x), Mathf.Round(coords[i].y + noiseShift.y));
        }
        
        return points;
    }
    public ColorEntry[] ColorDataLookUp(string objectType,bool useNoise)
    {   

        ColorEntry[] colorMap = null;

        bool colorMapExist = this.colorData.colorMaps.TryGetValue(objectType, out colorMap);
        if(!colorMapExist){return null;}

        if (useNoise){
            float totalPercent = 0f;
            //Noise is adjusted by ensuring that it will always sums to 100%
            //No value should be negative
            for (int i=0; i < colorMap.Length; i++)
            {
                //add noise at 2%
                colorMap[i].percent = colorMap[i].percent + Mathf.Abs(UnityEngine.Random.Range(-2,2)/100);
                if(colorMap[i].percent<=0){
                    //make sure the percentage is positive
                    colorMap[i].percent = 1e-4F;
                }
                totalPercent += colorMap[i].percent;
            }
            //To ensure total percentage sums up to 100
            for (int i =0; i < colorMap.Length; i++)
            {
                colorMap[i].percent = colorMap[i].percent/totalPercent;
            }
        }
        return colorMap;
    }

    //check out of bound vertices
    //naive method that needs to be refined 
    public bool IsOutOfBound(Vector2[] points)
    {
        float groundLevel = 0;
        if (GameObject.Find("Ground") != null){
            groundLevel = this.cam.WorldToScreenPoint(GameObject.Find("Ground").GetComponent<BoxCollider2D>().bounds.max).y;
        }

        //adjust origin to top-left
        //This is to gurantee the coordination system is consistent to the original chrome angry birds framework 
        groundLevel = Screen.height - groundLevel;

        int leftOutCount = 0;
        int rightOutCount = 0;
        int buttomOutCount = 0;
        int topOutCount = 0;

        for (int i = 0; i < points.Length; i++)
        {
            if (points[i].x <= 0)
            {
                leftOutCount += 1;
                if(leftOutCount >= points.Length){
                    return true;
                }
            }
            if (points[i].x >= Screen.width)
            {
                rightOutCount += 1;
                if(rightOutCount >= points.Length){
                    return true;
                }
            }
            if (points[i].y <= 0)
            {
                topOutCount += 1;
                if(topOutCount >= points.Length){
                    return true;
                }                
            }
            if (points[i].y >= Mathf.Min(Screen.height,groundLevel))
            {
                buttomOutCount += 1;
                if(buttomOutCount >= points.Length){
                    return true;
                }                
            }
        }
        return false;
    }


    //To obtain colors from Screenshot
    Texture2D TakeScreenshot()
    {
        Texture2D screenimage;

        screenimage = new Texture2D(Screen.width, Screen.height, TextureFormat.ARGB32, true);
        screenimage.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0, true);
        screenimage.Apply();

        //To calculate groundtruth for add_noise novelty, we use the unblurred/unflipped/unrotated screenshot(without representationnovelty)
        if (ABGameWorld.noveltyTypeForNovelty3 != 3)
        {
            screenimage = RepresentationNovelty.addNoveltyToScreenshotIfNeeded(screenimage);
        }
        screenimage = RepresentationNovelty.addNoveltyToScreenshotIfNeeded(screenimage);
        return screenimage;
    }

    public string GetScreenshotStr()
    {
        //yield return new WaitForEndOfFrame();
        Texture2D screenimage = TakeScreenshot();

        byte[] byteArray = screenimage.EncodeToPNG();
        return System.Convert.ToBase64String(byteArray);
    }

}