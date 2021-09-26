/**
* @author Peng Zhang
*
* @date - 04/08/2020 
*/

using UnityEngine;
using System.Collections.Generic;
/// <summary>   
/// The representation of one color from an object as the percent it occupied
/// </summary>
[System.Serializable]
public struct ColorEntry
{
    [SerializeField]
    public int color;
    [SerializeField]
    public float percent;
    public ColorEntry(int color, float percent)
    {
        this.color = color;
        this.percent = percent;
    }
}

/// <summary>   
/// The color map for a single object
/// </summary>   
[System.Serializable]
public class ColorMap
{
    //variables in the json file
    public string type;
    public Vector2[] colormap;

    public ColorEntry[] colorEntries;
    public void buildColorMap(){
        this.colorEntries = new ColorEntry[this.colormap.Length];
        for(int i = 0; i < this.colormap.Length; i++){
            this.colorEntries[i] = new ColorEntry((int)this.colormap[i].x,this.colormap[i].y);
        }
    }
}

/// <summary>   
/// Color data consists a list of color maps of all objects
/// </summary>   
public class ColorData
{
    //sprites in json object
    public ColorMap[] sprites;

    public Dictionary<string,ColorEntry[]> colorMaps = new Dictionary<string,ColorEntry[]>();
    public void readColorMaps(){
        foreach(ColorMap colorMap in sprites){
            colorMap.buildColorMap();
            colorMaps.Add(colorMap.type, colorMap.colorEntries);
        }
    }
}