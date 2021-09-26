/// <summary>
/// This class represents a science birds object in the ground truth symbolic game state  
/// </summary>

/**
* @author Peng Zhang
*
* @date - 04/08/2020 
*/
using UnityEngine;
using System.Collections.Generic;

public class GTGround : GTObjectBase{

    //The y coordinate of the ground level
    public int yindex {get; private set;}

    public GTGround(string id, string objectClass, int yindex):base(id,objectClass){
        this.yindex = yindex;
    }

    //implement the funtion that generates the json string for the object
    public override string ToJsonString(bool dev){
        
        string json = "{\"type\": \"Feature\",\"geometry\":{},";

        //porperty compoenet
        json += "\"properties\":{";
        //id
        json= json + "\"id\":\""+this.id+"\",";

        json = json +"\"label\":" +"\"Ground\",";

        //y coordinate of the ground line
        json= json + "\"yindex\":"+yindex+",";

        //color map
        json += "\"colormap\":";
        json += "[]";

        //end property component
        json+="}";

        //end object
        json+="}";
        
        return json;
    }
}