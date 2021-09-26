/// <summary>
/// This class represents a science birds object in the ground truth symbolic game state  
/// </summary>

/**
* @author Peng Zhang
*
* @date - 04/08/2020 
*/

using System.Collections.Generic;

public abstract class GTObjectBase{
    //the unique id of the object through its lifetime
    public readonly string id;
    //the class of the object, e.g. sling shot, ground or object
    // in development mode, precise classes of the objects are given, e.g. wood, ice, novel_object_1 etc.
    // for debugging purpose 
    public readonly string objectClass;
    public ColorEntry[] colorMap;
    public GTObjectBase(string id, string objectClass, ColorEntry[] colorMap = null){
        this.id = id;
        this.objectClass = objectClass;
        this.colorMap = colorMap;
    }

    //generate the json string of the object for the ground truth
    public abstract string ToJsonString(bool dev = false);
}