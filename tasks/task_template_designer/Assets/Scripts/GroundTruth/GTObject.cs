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

public class GTObject : GTObjectBase{
    
    //Each path of an object contains a list of ordered vertices of the object
    //As an object may be with holes or contains multiple parts, there may be more than one paths
    public List<Vector2[]> contours {get; private set;}

    public float currentLife;

    public GTObject(string id, string objectClass, ColorEntry[] colorMap, List<Vector2[]> paths, float currentLife):base(id,objectClass,colorMap){
        this.contours = paths;
        this.currentLife = currentLife;
    }

    //implement the funtion that generates the json string for the object
    public override string ToJsonString(bool dev){
        string json = "{\"type\": \"Feature\",\"geometry\":{";
        

        //geometric component
        //contours
        json = json +"\"type\":\"Polygon\",\"coordinates\":[";
        foreach(Vector2[] contour in this.contours){
            json+="[";
            foreach(Vector2 point in contour){
                json+="[";
                json += point.x;
                json+= ",";
                json += point.y;
                json+="],";
            }
            //remove last ,
            json = json.Remove(json.Length-1,1);
            json+="],";
        }
        //remove last ,
        json = json.Remove(json.Length-1,1);
        json+="]";

        json += "},";

        //end geometric component

        
        //porperty compoenet
        json += "\"properties\":{";
        //id
        json= json + "\"id\":\""+this.id+"\",";

        string type = this.objectClass;
        //type
        if(!dev){
            if(!type.Equals("Slingshot")){
                type = "Object";
            }   
        }
        json = json +"\"label\":" +"\""+type + "\",";

        //color map
        json += "\"colormap\":";
        json += "[";
        if(this.colorMap!=null){
            foreach(ColorEntry colorEntry in this.colorMap){
                int color = colorEntry.color;
                float percent = colorEntry.percent;
                json = json +"{" +"\"color\":"+color + ",\"percent\":" +percent + "},";
            }        
            //remove last ,
            json = json.Remove(json.Length-1,1);
        }

        json += "],";
        if(dev){
            json = json +"\"currentLife\":" + this.currentLife;
        }
        else{
            //remove last ,
            json = json.Remove(json.Length-1,1);
        }

        json+="}";//end property component

        json += "}";//end object

        return json;
    }
}