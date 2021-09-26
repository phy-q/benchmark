using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using SimpleJSON;
using System.Linq;
using System;

public class RepresentationNovelty : MonoBehaviour {


    public static string addNoveltyToJSONGroundTruthIfNeeded(string groundTruthJSONString) {
        Debug.Log("adding novelty level 3");
        switch (ABGameWorld.noveltyTypeForNovelty3) {
            case -1:
                // No novelty
                return groundTruthJSONString;

            case 1:
                // Grayscaling is done on image and ground truth colormap is updated as well
                // no extra action on ground truth is required here
                return groundTruthJSONString;

            case 2: 
                // Rotate world by 90 degrees (x-axis and y-axis reflection)
                return rotateAxis(90.0, groundTruthJSONString);
            
            case 3: 
                Debug.Log("rotating the scene by 180 degree");
                // Rotate world by 180 degrees (x-axis and y-axis reflection)
                return rotateAxis(180.0, groundTruthJSONString);

            case 4: 
                // Rotate world by 270 degrees (x-axis and y-axis reflection)
                return rotateAxis(270.0, groundTruthJSONString);

            default:
                return groundTruthJSONString;
        }
    }

    public static Texture2D addNoveltyToScreenshotIfNeeded(Texture2D screenshot) {
        switch (ABGameWorld.noveltyTypeForNovelty3) {
            case -1:
                // No novelty
                return screenshot;

            case 1:   
                // Grayscale Novelty
                return convertToGrayscale(screenshot);

            case 2:
                // Rotate by 90 degrees
                return rotateTexture(90, screenshot);

            case 3:
                // Rotate by 180 degrees
                return rotateTexture(180, screenshot);
            
            case 4:
                // Rotate by 270 degrees
                return rotateTexture(270, screenshot);
  
            default:
                return screenshot;
        }
    }

    public static double[] addNoveltyToActionSpaceIfNeeded(double x, double y, double dx, double dy) {
        
        switch(ABGameWorld.noveltyTypeForNovelty3) {
            case -1:
                // No novelty
                return new double[] { x, y, dx, dy};
            
            case 1:
                // No action novelty for grayscale novelty
                return new double[] { x, y, dx, dy};

            case 2:
                // In case world was rotated by 90 degrees, we need to rotate
                // action by 270 degrees to come back to original coordinates for simulator
                // here we expect the action to be received in the new coordinates
                return rotateAction(270.0, x, y, dx, dy);

            case 3:
                // In case world was rotated by 180 degrees, we need to rotate
                // action by 180 degrees to come back to original coordinates for simulator
                // here we expect the action to be received in the new coordinates
                return rotateAction(180.0, x, y, dx, dy);
            
            case 4:
                // In case world was rotated by 270 degrees, we need to rotate
                // action by 90 degrees to come back to original coordinates for simulator
                // here we expect the action to be received in the new coordinates
                return rotateAction(90.0, x, y, dx, dy);
                
            default:
                return new double[] { x, y, dx, dy};
        }
    }

    /**
     * Rotate image by given degree.
     * Because of the fact that image width and height are changing
     * when rotating, only multiples of 90 are supported for now.
    */
    private static Texture2D rotateTexture(double degree, Texture2D originalTexture)
    {   
        for (int i = 0; i < Math.Ceiling(degree / 90); i++) {
            originalTexture = rotateTexture(originalTexture, false);
        }
        
        return originalTexture;
    }

    private static Texture2D rotateTexture(Texture2D originalTexture, bool clockwise) {
        Color32[] original = originalTexture.GetPixels32();
        Color32[] rotated = new Color32[original.Length];
        int w = originalTexture.width;
        int h = originalTexture.height;
     
        int iRotated, iOriginal;
     
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                iRotated = (x + 1) * h - y - 1;
                iOriginal = clockwise ? original.Length - 1 - (y * w + x) : y * w + x;
                rotated[iRotated] = original[iOriginal];
            }
        }
     
        Texture2D rotatedTexture = new Texture2D(h, w);
        rotatedTexture.SetPixels32(rotated);
        rotatedTexture.Apply();
        return rotatedTexture;
    }


    private static int changeAxisOrigin(int axisPoint, int originShift) {
        return axisPoint + originShift;
    }

    private static double[] rotateAction(double degree, double x, double y, double dx, double dy) {

        if(degree == 90) {
            //x = x;
            y = (Screen.height) - y;
            //dx = dx;
            dy = (Screen.height) - dx;
        }
        else if(degree == 180) {
            x = (Screen.width) - x;
            y = (Screen.height) - y;
            dx = (Screen.width) - dx;
            dy = (Screen.height) - dx;
        }
        else if(degree == 270) {
            x = (Screen.width) - x;
            //y = y;
            dx = (Screen.width) - dx;
            //dy = dy;
        }

        return new double[] { x, y, dx, dy};
    }

    private static double[] rotateAction(double degree, double x, double y) {

        if(degree == 90) {
            //x = x;
            y = (Screen.height) - y;
        }
        else if(degree == 180) {
            x = (Screen.width) - x;
            y = (Screen.height) - y;
        }
        else if(degree == 270) {
            x = (Screen.width) - x;
            //y = y;
        }

        return new double[] { x, y};
    }


    private static string rotateAxis(double degree, string groundTruthJSONString) {
        
        JSONNode groundTruthJSON = JSONNode.Parse(groundTruthJSONString);
        
        foreach (JSONNode node in groundTruthJSON.Children) {
            // Debug.Log(string.Format("{0}", node.Value));
            
            foreach(JSONNode featureNode in node["features"].Children){
                if(featureNode["geometry"]["type"].Value.Equals("Polygon")){
                    foreach(JSONNode pathNode in featureNode["geometry"]["coordinates"].Children){
                        foreach(JSONNode pointNode in pathNode.Children){
                            JSONArray ptArray = pointNode.AsArray;
                            int x = ptArray[0].AsInt;
                            int y = ptArray[1].AsInt;
                            if(degree == 90) {
                                ptArray[0].AsInt = (Screen.height) - y;
                                ptArray[1].AsInt = x;
                            }
                            else if(degree == 180) {
                                ptArray[0].AsInt = (Screen.width) - x;
                                ptArray[1].AsInt = (Screen.height) - y;
                            }
                            else if(degree == 270) {
                                ptArray[0].AsInt = y;
                                ptArray[1].AsInt = (Screen.width) - x;
                            }
                                                                                                  
                        }
                    }

                }
                else if(featureNode["geometry"]["type"].Value.Equals("MultiPoint")){
                    foreach(JSONNode pointNode in featureNode["geometry"]["coordinates"].Children){
                        JSONArray ptArray = pointNode.AsArray;
                        int x = ptArray[0].AsInt;
                        int y = ptArray[1].AsInt;
                        if(degree == 90) {
                            ptArray[0].AsInt = (Screen.height) - y;
                            ptArray[1].AsInt = x;
                        }
                        else if(degree == 180) {
                            ptArray[0].AsInt = (Screen.width) - x;
                            ptArray[1].AsInt = (Screen.height) - y;
                        }
                        else if(degree == 270) {
                            ptArray[0].AsInt = y;
                            ptArray[1].AsInt = (Screen.width) - x;
                        }
                    }
                }   
            }
        }
        return Regex.Replace(groundTruthJSON.ToString(), @"(?<=\d+)(?<!""id"":""-\d+)(?<!""id"":""\d+)(?<!""type"":""\w+)(?<!""label"":""\w+)[""]|(?<!""type"":)(?<!""label"":)(?<!""id"":)[""]((?=\d+)|(?=-\d+))", "");
    }

    private static Texture2D convertToGrayscale(Texture2D graph)
    {
        Color32[] pixels = graph.GetPixels32();
        
        for (int x=0;x<graph.width;x++)
        {
            for (int y=0;y<graph.height;y++)
            {
                Color32 pixel = pixels[x+y*graph.width];
                int p =  ( (256 * 256 + pixel.r) * 256 + pixel.b) * 256 + pixel.g;
                int b = p % 256;
                p = Mathf.FloorToInt(p / 256);
                int g = p % 256;
                p = Mathf.FloorToInt (p / 256);
                int r = p % 256;
                float l = (0.2126f*r/255f) + 0.7152f*(g/255f) + 0.0722f*(b/255f);
                Color c = new Color(l,l,l,1);
                graph.SetPixel(x,y,c);
            }
        }
        graph.Apply(false);

        return graph;
    }

}
