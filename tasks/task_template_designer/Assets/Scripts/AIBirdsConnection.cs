// SCIENCE BIRDS: A clone version of the Angry Birds game used for 
// research purposes
// 
// Copyright (C) 2016 - Lucas N. Ferreira - lucasnfe@gmail.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>
//

using UnityEngine;
using System.Runtime.InteropServices;
using System.Collections;
using System.Collections.Generic;
using System;
using SimpleJSON;
using UnityEngine.SceneManagement;
using System.Diagnostics;
using System.IO;

delegate IEnumerator Handler(JSONNode data);


public class Message
{

    public string data;
    public string time;
}

public class AIBirdsConnection : ABSingleton<AIBirdsConnection>
{
    public int port;
    public bool isZoomingCompleted {get; private set;}

    Dictionary<String, Handler> handlers;
    WebSocket socket;
    static int dragCounter = 0;
    private bool ifRecord = true;
    string logPath;
    bool isPlaying;
    
    void log(string logText){
        if (File.Exists(logPath))
        {
            System.IO.File.AppendAllText(logPath, logText);
        }
        else
        {
            System.IO.File.WriteAllText(logPath, logText);
        }
    }
    IEnumerator Click(JSONNode data)
    {

        yield return new WaitForEndOfFrame();

        string id = data [0];

		Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        if(!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating||
        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest)){
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }
        else{
            float clickX = data[2]["x"].AsFloat;
            float clickY = Screen.height - data[2]["y"].AsFloat;

            Vector2 clickPos = new Vector2(clickX, clickY);

            HUD.Instance.SimulateInputEvent = 1;
            HUD.Instance.SimulateInputPos = clickPos;
            HUD.Instance.SimulateInputDelta = clickPos;

            id = data [0];

            msg = new Message();
            msg.data = "1";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }

    #if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));	
	#else
		socket.Send(message);	
	#endif
	}

	IEnumerator Drag(JSONNode data) {
	
		yield return new WaitForEndOfFrame ();

        string id = data [0];

        Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        if((!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating))||
        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest){
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }
        else{
            LoadLevelSchema.Instance.nInteractions++;
            LoadLevelSchema.Instance.nOverallInteractions++;
            UnityEngine.Debug.Log("number of interactions " + LoadLevelSchema.Instance.nInteractions);
            
            LoadLevelSchema.Instance.updateLikelihoodRequestFlag();
            dragCounter++;
            float dragX = data[2]["x"].AsFloat;
            float dragY = data[2]["y"].AsFloat;

            float dragDX = dragX + data[2]["dx"].AsFloat;
            float dragDY = dragY + data[2]["dy"].AsFloat;
            
            float tapTime = data[2]["tap_time"].AsFloat;
            UnityEngine.Debug.Log ("TAP = " + tapTime);
            // Scale tapTime by time 
            tapTime = tapTime / ABGameWorld.SimulationSpeed;

            

            Vector2 dragPos = new Vector2 (dragX, Screen.height - dragY);
            Vector2 deltaPos = new Vector2 (dragDX, Screen.height - dragDY);
            UnityEngine.Debug.Log ("POS = " + dragPos);
            UnityEngine.Debug.Log ("DRAG = " + deltaPos);

            RepresentationNovelty.addNoveltyToActionSpaceIfNeeded(dragX, dragY, dragDX, dragDY);
            HUD.Instance.shootDone = false;
            HUD.Instance.SimulateInputEvent = 1;
            HUD.Instance.SimulateInputPos = dragPos;
            HUD.Instance.SimulateInputDelta = deltaPos;
            HUD.Instance.SimulatedTapTime = tapTime;

            if (ifRecord)
            {
                string recordingText = "DragDetails:" + DateTime.Now.ToString("yyyyMMddHHmmss") + ": levelName :" + LoadLevelSchema.Instance.currentLevelPaths[LevelList.Instance.CurrentIndex] + ": trial :" + LoadLevelSchema.Instance.currentTrialIndex + ": dragx :" + dragX + ": dragy :" + dragY + ": dragDX :" + dragDX + ": DragDY :" + dragDY + ": TapTime :"+ tapTime+": Score :" + HUD.Instance.GetScore() + ",end" + "\n";

                log(recordingText);
            }

            id = data [0];

            msg = new Message();
            msg.data = "1";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";



            while(!HUD.Instance.shootDone){
                UnityEngine.Debug.Log("waiting for the shot to be completed. shootDone is " + HUD.Instance.shootDone);
                UnityEngine.Debug.Log("mouse sim step " + HUD.Instance.SimulateInputEvent);
                if (ABGameWorld.Instance.LevelCleared () || ABGameWorld.Instance.LevelFailed ()) {
                    HUD.Instance.SimulateInputEvent = 0;
                    HUD.Instance.shootDone = true;
                }
                yield return new WaitForEndOfFrame ();
            } 
        }
		
	#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator TapShoot(JSONNode data) {
	
		yield return new WaitForEndOfFrame ();

        string id = data [0];

        Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        if((!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating))||
        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest){
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }
        else{
            LoadLevelSchema.Instance.nInteractions++;
            LoadLevelSchema.Instance.nOverallInteractions++;
            UnityEngine.Debug.Log("number of interactions " + LoadLevelSchema.Instance.nInteractions);
            
            Vector3 slingshotPos = ABGameWorld.Instance.Slingshot().transform.position - ABConstants.SLING_SELECT_POS;
            Vector3 slingScreenPos = Camera.main.WorldToScreenPoint(slingshotPos);

            LoadLevelSchema.Instance.updateLikelihoodRequestFlag();
            dragCounter++;
            float dragX = slingScreenPos.x;
            float dragY = slingScreenPos.y;

            float dragDX = data[2]["x"].AsFloat;
            float dragDY = data[2]["y"].AsFloat;
            
            float tapTime = data[2]["tap_time"].AsFloat;
            UnityEngine.Debug.Log ("TAP = " + tapTime);
            // Scale tapTime by time 
            tapTime = tapTime / ABGameWorld.SimulationSpeed;

            
            //no need to convert the coordinate of the drag pt since it is from the unity ingame coordinate but not from the agent 
            Vector2 dragPos = new Vector2 (dragX,dragY);
            //this one is from the agent so it still needs to be converted
            Vector2 deltaPos = new Vector2 (dragDX, Screen.height - dragDY);
            
            UnityEngine.Debug.Log ("POS = " + dragPos);
            UnityEngine.Debug.Log ("DRAG = " + deltaPos);

            RepresentationNovelty.addNoveltyToActionSpaceIfNeeded(dragX, dragY, dragDX, dragDY);
            HUD.Instance.shootDone = false;
            HUD.Instance.SimulateInputEvent = 1;

            //for one tap shoot without dragging, this should be the origin point of the bird on the sling  
            HUD.Instance.SimulateInputPos = dragPos;
            
            //then this should come from the agent's request
            HUD.Instance.SimulateInputDelta = deltaPos;
            
            
            HUD.Instance.SimulatedTapTime = tapTime;

            if (ifRecord)
            {
                string recordingText = "DragDetails:" + DateTime.Now.ToString("yyyyMMddHHmmss") + ": levelName :" + LoadLevelSchema.Instance.currentLevelPaths[LevelList.Instance.CurrentIndex] + ": trial :" + LoadLevelSchema.Instance.currentTrialIndex + ": dragx :" + dragX + ": dragy :" + dragY + ": dragDX :" + dragDX + ": DragDY :" + dragDY + ": TapTime :"+ tapTime+": Score :" + HUD.Instance.GetScore() + ",end" + "\n";

                log(recordingText);
            }

            id = data [0];

            msg = new Message();
            msg.data = "1";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";



            while(!HUD.Instance.shootDone){
                UnityEngine.Debug.Log("waiting for the shot to be completed. shootDone is " + HUD.Instance.shootDone);
                UnityEngine.Debug.Log("mouse sim step " + HUD.Instance.SimulateInputEvent);
                if (ABGameWorld.Instance.LevelCleared () || ABGameWorld.Instance.LevelFailed ()) {
                    HUD.Instance.SimulateInputEvent = 0;
                    HUD.Instance.shootDone = true;
                }
                yield return new WaitForEndOfFrame ();
            } 
        }
		
	#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator ShootAndRecordGroundTruth(JSONNode data) {
        
        yield return new WaitForEndOfFrame ();

        UnityEngine.Debug.Log("Received Shoot and Record request.");

        string id = data [0];

		Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message_i = "[" + id + "," + json + "]";

        if((!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating))||
        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest){
            UnityEngine.Debug.Log("Not allowed to shoot.");
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message_i = "[" + id + "," + json + "]";

            #if UNITY_WEBGL && !UNITY_EDITOR
		        socket.Send(System.Text.Encoding.UTF8.GetBytes(message_i));
	        #else
		        socket.Send(message_i);	
	        #endif

        }
        else{
            LoadLevelSchema.Instance.nInteractions++;
            LoadLevelSchema.Instance.nOverallInteractions++;
            UnityEngine.Debug.Log("number of interactions " + LoadLevelSchema.Instance.nInteractions);
            
            Vector3 slingshotPos = ABGameWorld.Instance.Slingshot().transform.position - ABConstants.SLING_SELECT_POS;
            Vector3 slingScreenPos = Camera.main.WorldToScreenPoint(slingshotPos);

            LoadLevelSchema.Instance.updateLikelihoodRequestFlag();
            dragCounter++;
            float dragX = slingScreenPos.x;
            float dragY = slingScreenPos.y;

            float dragDX = data[2]["x"].AsFloat;
            float dragDY = data[2]["y"].AsFloat;
            
            float tapTime = data[2]["tap_time"].AsFloat;

            int groundTruthTakeFrequency = data[2]["gt_freq"].AsInt;

            UnityEngine.Debug.Log ("TAP = " + tapTime);
            
            ABGameWorld.takeGroundTruthEveryNthFrames = groundTruthTakeFrequency;
            // Scale tapTime by time 
            tapTime = tapTime / ABGameWorld.SimulationSpeed;

            Vector2 dragPos = new Vector2 (dragX, dragY);
            Vector2 deltaPos = new Vector2 (dragDX, Screen.height - dragDY);
            UnityEngine.Debug.Log ("POS = " + dragPos);
            UnityEngine.Debug.Log ("DRAG = " + deltaPos);

            RepresentationNovelty.addNoveltyToActionSpaceIfNeeded(dragX, dragY, dragDX, dragDY);
            HUD.Instance.shootDone = false;
            HUD.Instance.SimulateInputEvent = 1;
            HUD.Instance.SimulateInputPos = dragPos;
            HUD.Instance.SimulateInputDelta = deltaPos;
            HUD.Instance.SimulatedTapTime = tapTime;
            // Start recording
            ABGameWorld.isRecordingBatchGroundTruth = true;

            id = data [0];

            // Wait until shot is over
            UnityEngine.Debug.Log("Waiting for a shot to be over...");

            while(!HUD.Instance.shootDone){
                UnityEngine.Debug.Log("waiting for the shot to be completed. shootDone is " + HUD.Instance.shootDone);
                UnityEngine.Debug.Log("mouse sim step " + HUD.Instance.SimulateInputEvent);
                if (ABGameWorld.Instance.LevelCleared () || ABGameWorld.Instance.LevelFailed ()) {
                    HUD.Instance.SimulateInputEvent = 0;
                    HUD.Instance.shootDone = true;
                }
                yield return new WaitForEndOfFrame ();
            } 

            UnityEngine.Debug.Log("Shot is done, waiting for ground truth to record on background thread");

            StartCoroutine(sendBatchGroundTruthUponCompletion(id, false));
        }	

    }

    private float MAX_TOTAL_WAIT_TIME_FOR_BATCH = 120.0f; // 2 minutes

    IEnumerator sendBatchGroundTruthUponCompletion(string id, bool shouldSendImage) {

        UnityEngine.Debug.Log("Starting recording GT on background thread...");

        Message msg = new Message();
        msg.data = "0";
        msg.time = DateTime.Now.ToString();

        string json_i = JsonUtility.ToJson(msg);

        string message_i = "[" + id + "," + json_i + "]";

        float totalWaitTime = 0.0f; // plan B, if something went wrong stop waiting

        while(ABGameWorld.isRecordingBatchGroundTruth == true && ABGameWorld.wasBirdLaunched == true) {
            UnityEngine.Debug.Log("Waiting.......");
            totalWaitTime += 1.0f;
            if(totalWaitTime > MAX_TOTAL_WAIT_TIME_FOR_BATCH) {
                UnityEngine.Debug.Log("Gave up waiting.......");
                break;
            }
            yield return new WaitForSeconds(1.0f);
        }

        try {
            UnityEngine.Debug.Log("Sending ground truth.");

            if(ABGameWorld.batchGroundTruths.Count == 0) {
                UnityEngine.Debug.Log("Empty batch of ground truths was received, return current state.");
                bool useNoise = LoadLevelSchema.Instance.devMode; 
                SymbolicGameState gt = new SymbolicGameState(useNoise);
                string gtjson = gt.GetGTJson();
                string image = gt.GetScreenshotStr();

                ABGameWorld.batchGroundTruths.Add((image, gtjson));
            }
            // send all ground truth
            UnityEngine.Debug.Log("Sending ground truths, count: " + ABGameWorld.batchGroundTruths.Count);
            msg.data = "data:image/png;base64," + ABGameWorld.batchGroundTruths[0].Item1;
            msg.time = DateTime.Now.ToString();

            json_i = JsonUtility.ToJson(msg);
            if (shouldSendImage == true){
                // Keep this functionality for us and future, but to save on data consumption turn it off by default
                message_i = "[" + id + "," + ABGameWorld.batchGroundTruths.Count + "," + 0 + "," + json_i + "," + ABGameWorld.batchGroundTruths[0].Item2.Substring(1, ABGameWorld.batchGroundTruths[0].Item2.Length - 1);
            } else {
                message_i = "[" + id + "," + ABGameWorld.batchGroundTruths.Count + "," + 0 + "," + ABGameWorld.batchGroundTruths[0].Item2.Substring(1, ABGameWorld.batchGroundTruths[0].Item2.Length - 1);
            }
            for(int i = 1; i < ABGameWorld.batchGroundTruths.Count; i++) {
                (string, string) ground_truth_i = ABGameWorld.batchGroundTruths[i];

                Message msg_i = new Message();
                msg_i.data = "data:image/png;base64," + ground_truth_i.Item1;
                msg_i.time = DateTime.Now.ToString();

                string json_internal = JsonUtility.ToJson(msg_i);  
        
                // for the first frame only send how many ground truths there are to expect
                if (shouldSendImage == true){
                    // Keep this functionality for us and future, but to save on data consumption turn it off by default
                    message_i += "|GTEND|" + "[" + id + "," + ABGameWorld.batchGroundTruths.Count + "," + i + "," + json_internal + "," + ground_truth_i.Item2.Substring(1, ground_truth_i.Item2.Length - 1);
                } else {
                    message_i += "|GTEND|" + "[" + id + "," + ABGameWorld.batchGroundTruths.Count + "," + i + "," + ground_truth_i.Item2.Substring(1, ground_truth_i.Item2.Length - 1);
                } 
            }

            #if UNITY_WEBGL && !UNITY_EDITOR
                socket.Send(System.Text.Encoding.UTF8.GetBytes(message_i));
            #else
                socket.Send(message_i);	
            #endif

            ABGameWorld.batchGroundTruths.Clear();
            ABGameWorld.batchGroundTruths = new List<(string, string)>(); // clear the list    
            UnityEngine.Debug.Log("Done sending ground truth.");    
        } catch (Exception ex) {
            UnityEngine.Debug.Log(ex.ToString());
        }
    }


    IEnumerator MouseWheel(JSONNode data)
    {
        yield return new WaitForEndOfFrame();

        string id = data [0];

		Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        if((!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating))||
        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest){
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }
        else{
            float delta = data[2]["delta"].AsFloat;

            //in case the camera object is null
            try{

                HUD.Instance.CameraZoom(-delta);

                id = data [0];

                msg = new Message();
                msg.data = "1";
                msg.time = DateTime.Now.ToString();

                json = JsonUtility.ToJson(msg);

                message = "[" + id + "," + json + "]";
            }

            catch{
                UnityEngine.Debug.LogWarning("zooming the camear failed, check if camear object is null");
                id = data [0];

                msg = new Message();
                msg.data = "0";
                msg.time = DateTime.Now.ToString();

                json = JsonUtility.ToJson(msg);

                message = "[" + id + "," + json + "]";
            }


        }


#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator Screenshot(JSONNode data)
    {
        yield return new WaitForEndOfFrame();
        SymbolicGameState gt = new SymbolicGameState();

        string image = gt.GetScreenshotStr();

        string id = data[0];

        Message msg = new Message();
        msg.data = "data:image/png;base64," + image;
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        if (ifRecord)
        {
            string level = "";
            string trial = "";
            try
            {
                level = LoadLevelSchema.Instance.currentLevelPaths[LevelList.Instance.CurrentIndex];
                trial = LoadLevelSchema.Instance.currentTrialIndex.ToString();
            }
            catch (System.Exception)
            {
                level = "na";
                trial = "na";
            }
            string recordingText = "ScreenShotRequest:" + DateTime.Now.ToString("yyyyMMddHHmmss") + ": levelName :" + level + ": trial :" + trial + ",end" + "\n";
            log(recordingText);
        }

        string message = "[" + id + "," + json + "]";


#if UNITY_WEBGL && !UNITY_EDITOR
		        socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator GroundTruthWithoutScreenshot(JSONNode data)
    {
        yield return new WaitForEndOfFrame();
        while(!this.isZoomingCompleted){
            yield return new WaitForEndOfFrame ();
        }
        
        string id = data[0];
        bool devMode = LoadLevelSchema.Instance.devMode;
        bool useNoise = !devMode;//do not apply noise if in dev mode 
        SymbolicGameState gt = new SymbolicGameState(useNoise);
        
        string gtjson = gt.GetGTJson();
       
        if (ifRecord)
        {
            string level = "";
            string trial = "";
            try
            {
                level = LoadLevelSchema.Instance.currentLevelPaths[LevelList.Instance.CurrentIndex];
                trial = LoadLevelSchema.Instance.currentTrialIndex.ToString();
            }
            catch (System.Exception)
            {
                level = "na";
                trial = "na";
            }
            string recordingText = "GTwithoutSS:" + DateTime.Now.ToString("yyyyMMddHHmmss") + ": levelName :" + level + ": trial :" + trial+ ",end" + "\n";
            //string recordingText = "GTWithoutSS:" + DateTime.Now.ToString("yyyyMMddHHmmss") + ",end" + "\n";
            log(recordingText);
        }
        
        string message = "[" + id + "," + gtjson.Substring(1, gtjson.Length - 1);

#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
        
    }

    IEnumerator NoisyGroundTruthWithoutScreenshot(JSONNode data)
    {
        yield return new WaitForEndOfFrame();
        while(!this.isZoomingCompleted){
            yield return new WaitForEndOfFrame ();
        }
        string id = data[0];

        SymbolicGameState gt = new SymbolicGameState(true);//apply noise
        
        string gtjson = gt.GetGTJson();

        if (ifRecord)
        {
            string level = "";
            string trial = "";
            try
            {
                level = LoadLevelSchema.Instance.currentLevelPaths[LevelList.Instance.CurrentIndex];
                trial = LoadLevelSchema.Instance.currentTrialIndex.ToString();
            }
            catch (System.Exception)
            {
                level = "na";
                trial = "na";
            }
            string recordingText = "NoisyGTwithoutSS:" + DateTime.Now.ToString("yyyyMMddHHmmss") + ": levelName :" + level + ": trial :" + trial + ",end" + "\n";
            log(recordingText);
        }

        string message = "[" + id + "," + gtjson.Substring(1, gtjson.Length - 1);

#if UNITY_WEBGL && !UNITY_EDITOR
		                socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    IEnumerator GroundTruthWithScreenshot(JSONNode data)
    {
        
        yield return new WaitForEndOfFrame();
        while(!this.isZoomingCompleted){
            yield return new WaitForEndOfFrame ();
        }
        string id = data[0];
        bool devMode = LoadLevelSchema.Instance.devMode;
        bool useNoise = !devMode;//do not apply noise if in dev mode 

        SymbolicGameState gt = new SymbolicGameState(useNoise);
        string gtjson = gt.GetGTJson();
        string image = gt.GetScreenshotStr();

        Message msg = new Message();
        msg.data = "data:image/png;base64," + image;
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        if (ifRecord)
        {
            string level = "";
            string trial = "";
            try
            {
                level = LoadLevelSchema.Instance.currentLevelPaths[LevelList.Instance.CurrentIndex];
                trial = LoadLevelSchema.Instance.currentTrialIndex.ToString();
            }
            catch (System.Exception)
            {
                level = "na";
                trial = "na";
            }
            string recordingText = "GTwithSS:" + DateTime.Now.ToString("yyyyMMddHHmmss") + ": levelName :" + level + ": trial :" + trial + ",end" + "\n";
            log(recordingText);
        }

        string message = "[" + id + "," + json + "," + gtjson.Substring(1, gtjson.Length - 1);

#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator NoisyGroundTruthWithScreenshot(JSONNode data)
    {
        yield return new WaitForEndOfFrame();
        while(!this.isZoomingCompleted){
            yield return new WaitForEndOfFrame ();
        }
        string id = data[0];

        SymbolicGameState gt = new SymbolicGameState(true);//apply noise
        string gtjson = gt.GetGTJson();
        string image = gt.GetScreenshotStr();

        Message msg = new Message();
        msg.data = "data:image/png;base64," + image;
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        if (ifRecord)
        {
            string level = "";
            string trial = "";
            try
            {
                level = LoadLevelSchema.Instance.currentLevelPaths[LevelList.Instance.CurrentIndex];
                trial = LoadLevelSchema.Instance.currentTrialIndex.ToString();
            }
            catch (System.Exception)
            {
                level = "na";
                trial = "na";
            }
            string recordingText = "NoisyGTwithSS:" + DateTime.Now.ToString("yyyyMMddHHmmss") + ": levelName :" + level + ": trial :" + trial + ",end" + "\n";
            log(recordingText);
        }

        string message = "[" + id + "," + json + "," + gtjson.Substring(1, gtjson.Length - 1);

#if UNITY_WEBGL && !UNITY_EDITOR
		        socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    IEnumerator SelectNextAvailableLevel(JSONNode data){
        yield return new WaitForEndOfFrame();
        isZoomingCompleted = false;
        string id = data [0];

		Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        if(!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating||
        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest)){
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }

        else{
            TrialInfo trial = LoadLevelSchema.Instance.trials[LoadLevelSchema.Instance.currentTrialIndex];
            GameLevelSetInfo gameLevelSet;
            if(LoadLevelSchema.Instance.isTesting){
                gameLevelSet = trial.testLevelSets[trial.currentTestSetIndex];
            }
            else{
                gameLevelSet = trial.trainingSet;
            }
            if(isPlaying){//if skip levels while the level is not finished
                EvaluationHandler.Instance.RecordEvaluationScore("Playing");
            }
            if(gameLevelSet.availableLevelList.Count>0){

                int currentLevelIndex = LevelList.Instance.CurrentIndex;
                LinkedListNode<int> currentNode;
                LinkedListNode<int> nextNode;
                if(currentLevelIndex < 0){
                    currentLevelIndex = 0;
                }
                //repeat the level is the current level's allowed number of attempts is greater than zero 
                if(gameLevelSet.nLevelEntryList[currentLevelIndex]>0){
                    nextNode = gameLevelSet.availableLevelList.Find(currentLevelIndex);
                }

                else{
                    if(currentLevelIndex>=0){
                        currentNode = gameLevelSet.availableLevelList.Find(currentLevelIndex);
                        nextNode = currentNode.Next;
                    } 
                    else{//if the levelindex is inited to -1, read the first level
                        currentNode = gameLevelSet.availableLevelList.First;
                        nextNode = currentNode; 
                    }
                }
                if(nextNode == null){
                    nextNode = gameLevelSet.availableLevelList.First;
                }

                //delayed remove to prevent current node to be null       
                if(gameLevelSet.nLevelEntryList[currentLevelIndex] == 0){
                    gameLevelSet.availableLevelList.Remove(currentLevelIndex);
                }

                int levelIndex = nextNode.Value;

                UnityEngine.Debug.Log("Level index:" + levelIndex);

                LevelList.Instance.SetLevel(levelIndex);
                gameLevelSet.nLevelEntryList[levelIndex]--;


                //do not need delayed remove if only one level is left
                if(gameLevelSet.nLevelEntryList[levelIndex] == 0 && gameLevelSet.availableLevelList.Count == 1){
                    gameLevelSet.availableLevelList.Remove(levelIndex);
                }

                UnityEngine.Debug.Log("loading next available level " + levelIndex.ToString());
                Time.timeScale = ABGameWorld.SimulationSpeed;
                ABSceneManager.Instance.LoadScene("GameWorld");

                id = data [0];

                msg = new Message();
                int returnValue = nextNode.Value + 1; 
                msg.data = returnValue.ToString();
                msg.time = DateTime.Now.ToString();

                json = JsonUtility.ToJson(msg);

                message = "[" + id + "," + json + "]";

                var currentScence = SceneManager.GetActiveScene().name;
                while(currentScence != "GameWorld"){
                    currentScence = SceneManager.GetActiveScene().name;
                    UnityEngine.Debug.Log("wait for level loading " + levelIndex.ToString());  
                    UnityEngine.Debug.Log("game scene " + currentScence);              
                    yield return new WaitForSeconds(1);
                }
                yield return new WaitForSeconds(2);
                try{
                    HUD.Instance.CameraZoom(15f);
                }
                catch{
                    UnityEngine.Debug.LogWarning("zooming out after loading level failed, check if camera object is null");
                }

                isZoomingCompleted = true;

                LevelList.Instance.CurrentIndex = levelIndex;
                LoadLevelSchema.Instance.nInteractions++;
                LoadLevelSchema.Instance.nOverallInteractions++;
                LoadLevelSchema.Instance.updateLikelihoodRequestFlag();
            }

            else{
                UnityEngine.Debug.Log("loading next available level : no available level");
                trial.loadLevelAfterFinished = true;
                id = data [0];

                msg = new Message();
                msg.data = "0";
                msg.time = DateTime.Now.ToString();

                json = JsonUtility.ToJson(msg);

                message = "[" + id + "," + json + "]";
            }

        }

#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator SelectLevel(JSONNode data)
    {

        yield return new WaitForEndOfFrame();

        string id = data [0];

		Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        if(!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating||
        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest)){
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }

        else{
            TrialInfo trial = LoadLevelSchema.Instance.trials[LoadLevelSchema.Instance.currentTrialIndex];
            GameLevelSetInfo gameLevelSet;
            if(LoadLevelSchema.Instance.isTesting){
                gameLevelSet = trial.testLevelSets[trial.currentTestSetIndex];
            }
            else{
                gameLevelSet = trial.trainingSet;
            }
            if(isPlaying){//if skip levels while the level is not finished
                EvaluationHandler.Instance.RecordEvaluationScore("Playing");
            }

            if(gameLevelSet.availableLevelList.Count>0){


                if(LoadLevelSchema.Instance.isSequence){
                    
                    int currentLevelIndex = LevelList.Instance.CurrentIndex;
                    //availableLevelList start from 1
                    LinkedListNode<int> currentNode = gameLevelSet.availableLevelList.Find(currentLevelIndex);
                    LinkedListNode<int> nextNode = currentNode.Next; 
                    if(nextNode == null){
                        nextNode = gameLevelSet.availableLevelList.First;
                    }

                    if(gameLevelSet.nLevelEntryList[currentLevelIndex] == 0){
                        gameLevelSet.availableLevelList.Remove(currentLevelIndex);
                    }

                    int levelIndex = nextNode.Value;
                    int requestedLevelIndex = data[2]["levelIndex"].AsInt;
                    if(levelIndex != (requestedLevelIndex-1)){
                        id = data [0];

                        msg = new Message();
                        msg.data = "0";
                        msg.time = DateTime.Now.ToString();

                        json = JsonUtility.ToJson(msg);

                        message = "[" + id + "," + json + "]";
                        
                    }  
                    else {

                        LevelList.Instance.SetLevel(requestedLevelIndex - 1);
                        gameLevelSet.nLevelEntryList[levelIndex]--;
                        if(gameLevelSet.nLevelEntryList[levelIndex] == 0 && gameLevelSet.availableLevelList.Count == 1){
                            gameLevelSet.availableLevelList.Remove(levelIndex);
                        }
                        Time.timeScale = ABGameWorld.SimulationSpeed;
                        ABSceneManager.Instance.LoadScene("GameWorld");
                        LoadLevelSchema.Instance.nInteractions++;
                        LoadLevelSchema.Instance.nOverallInteractions++;
                        LoadLevelSchema.Instance.updateLikelihoodRequestFlag();
                        id = data [0];

                        msg = new Message();
                        msg.data = "1";
                        msg.time = DateTime.Now.ToString();

                        json = JsonUtility.ToJson(msg);

                        message = "[" + id + "," + json + "]";
                        UnityEngine.Debug.Log("Level index:" + levelIndex);

                    }
                

                }
                else{
                    int requestedLevelIndex = data[2]["levelIndex"].AsInt;
                    if(gameLevelSet.availableLevelList.Find(requestedLevelIndex)!=null){

                        UnityEngine.Debug.Log("Level index:" + requestedLevelIndex);

                        LevelList.Instance.SetLevel(requestedLevelIndex - 1);
                        gameLevelSet.nLevelEntryList[requestedLevelIndex - 1]--;
                        if(gameLevelSet.nLevelEntryList[requestedLevelIndex - 1] == 0){
                            gameLevelSet.availableLevelList.Remove(requestedLevelIndex - 1);
                        }   

                        Time.timeScale = ABGameWorld.SimulationSpeed;
                        ABSceneManager.Instance.LoadScene("GameWorld");
                        LoadLevelSchema.Instance.nInteractions++;
                        LoadLevelSchema.Instance.nOverallInteractions++;
                        LoadLevelSchema.Instance.updateLikelihoodRequestFlag();
                        id = data [0];

                        msg = new Message();
                        msg.data = "1";
                        msg.time = DateTime.Now.ToString();

                        json = JsonUtility.ToJson(msg);

                        message = "[" + id + "," + json + "]";
                    }

                    else{

                        id = data [0];

                        msg = new Message();
                        msg.data = "0";
                        msg.time = DateTime.Now.ToString();

                        json = JsonUtility.ToJson(msg);

                        message = "[" + id + "," + json + "]";
                    }

                }

            }

            else{
                trial.loadLevelAfterFinished = true;
                id = data [0];
                msg = new Message();
                msg.data = "0";
                msg.time = DateTime.Now.ToString();

                json = JsonUtility.ToJson(msg);

                message = "[" + id + "," + json + "]";

            }
        }
#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator LoadScene(JSONNode data)
    {

        yield return new WaitForEndOfFrame();

        string id = data [0];

		Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        if(!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating||
        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest)){
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }
        else{
            string scene = data[2]["scene"];

            Time.timeScale = ABGameWorld.SimulationSpeed;
            ABSceneManager.Instance.LoadScene(scene);

            id = data [0];

            msg = new Message();
            msg.data = "1";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }
#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator Score(JSONNode data)
    {

        yield return new WaitForEndOfFrame();
        var currentScence = SceneManager.GetActiveScene().name;

        string id = data[0];
        
        Message msg = new Message();
        msg.data = "0";
        if(currentScence == "GameWorld"){
            msg.data = HUD.Instance.GetScore().ToString();
        }
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);
        string message = "[" + id + "," + json + "]";

#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    IEnumerator NoveltyInfo(JSONNode data){
        yield return new WaitForEndOfFrame();
        string id = data[0];
        string json;
        string message;

        Message msg = new Message();
        TrialInfo trial = LoadLevelSchema.Instance.trials[LoadLevelSchema.Instance.currentTrialIndex];
        if(trial.notifyNovelty){
            string currentLevelPath; 
            string[] currentLevelPathArray = LoadLevelSchema.Instance.currentLevelPaths;
            int currentLevelIndex = LevelList.Instance.CurrentIndex;
            currentLevelPath = currentLevelPathArray[currentLevelIndex];

            string[] stringSeparators = new string[] {"novelty_level_"};
            string noveltyLevelString = currentLevelPath.Split (stringSeparators, StringSplitOptions.None)[1].Substring(0,1);
            int noveltyLevel = int.Parse(noveltyLevelString);

            if(noveltyLevel > 0){
                msg.data = "1";
            }
            else {
                msg.data = "0";
            }

        }
        else{
            msg.data = "-1";

        }
        msg.time = DateTime.Now.ToString();
        json = JsonUtility.ToJson(msg);
        message = "[" + id + "," + json + "]";


#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif

    }

    IEnumerator GameState(JSONNode data)
    {

        yield return new WaitForEndOfFrame();
        string id = data[0];

        string currentScence;

        string json;
        string message;
        Message msg;
        
        currentScence = SceneManager.GetActiveScene().name;

        id = data[0];

        if (currentScence == "GameWorld")
        {

            if (ABGameWorld.Instance.LevelCleared())
            {

                currentScence = "LevelCleared";
            }
            else if (ABGameWorld.Instance.LevelFailed())
            {

                currentScence = "LevelFailed";
            }
            else if (!ABGameWorld.Instance.IsLevelStable())
            {

                currentScence = "LevelUnstable";
            }
        }

        //wait until a stable state to update the evaluation related states
        //if(currentScence != "LevelUnstable"){
        
        //report novelty likelihood has priority
        if(LoadLevelSchema.Instance.noveltyLikelihoodReportRequest){
            currentScence = "NoveltyLikelihoodReportRequest";
        }
        else if(LoadLevelSchema.Instance.isNewTrial){                
            currentScence = "NewTrial";
        }

        else if(LoadLevelSchema.Instance.evaluationTerminating){
            currentScence = "EvaluationTerminated";
        }      
        //wait until the level is solved 
        //GameWorld means playing 
        else if(currentScence != "LevelUnstable" && currentScence != "GameWorld"){

            //ask the agent to report novelty likelihood first, before any new set/termination states
            if(LoadLevelSchema.Instance.isCheckPoint){   
                if(LoadLevelSchema.Instance.isTesting){
                    currentScence = "NewTestSet";
                }
                else{
                    currentScence = "NewTrainingSet";
                }
            }

            else if(LoadLevelSchema.Instance.resumeTraining){
                currentScence = "ResumeTraining";
            }
        }
        msg = new Message();
        msg.data = currentScence;
        msg.time = DateTime.Now.ToString();
        json = JsonUtility.ToJson(msg);
        message = "[" + id + "," + json + "]";


#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    IEnumerator NoOfLevels(JSONNode data)
    {

        yield return new WaitForEndOfFrame();
        while(!LoadLevelSchema.Instance.initializationDone){
            yield return new WaitForEndOfFrame();            
        }
        string id = data[0];

        Message msg = new Message();
        msg.data = LevelList.Instance.GetNumberOfLevels().ToString();
        msg.time = DateTime.Now.ToString();

        UnityEngine.Debug.Log("Num Of Levels: " + msg.data);

        string json = JsonUtility.ToJson(msg);
        string message = "[" + id + "," + json + "]";

#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    IEnumerator SpeedUpGameSimulation(JSONNode data)
    {
        yield return new WaitForEndOfFrame();

        float simSpeed = data[2]["sim_speed"].AsFloat;

        UnityEngine.Debug.Log("SIM_SPEED = " + simSpeed);

        // Make sure speed is within Unity boundaries 0-50
        //String optionalError = "{}";
        if (simSpeed > 50.0f)
        {
            simSpeed = 50.0f;
        //    optionalError = "{\"error\": \"Simulation speed must be within 0.0 to 50.0.\"}";
        }
        else if (simSpeed <= 0.0f)
        {
            simSpeed = 0.00001f; // setting speed to 0 will freeze the game completely
        //    optionalError = "{\"error\": \"Simulation speed must be within 0.00001 to 50.0.\"}";
        }

        ABGameWorld.SimulationSpeed = simSpeed;

        string id = data[0];
        Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);
        string message = "[" + id + "," + json + "]";

#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    IEnumerator ReportNoveltyLikelihood(JSONNode data)
    {
        yield return new WaitForEndOfFrame();

        LoadLevelSchema.Instance.noveltyLikelihoodReportRequest = false;
        string id = data [0];

		Message msg = new Message();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        float noveltyLikelihood = data[2]["novelty_likelihood"].AsFloat;
        float nonNoveltyLikelihood = data[2]["non_novelty_likelihood"].AsFloat;

        JSONArray novelIds = data[2]["object_ids"].AsArray;

        // Deal with the case of a non-array value.
        if (novelIds == null) { /*...*/ }

        // Create an int array to accomodate the numbers.
        int[] novelIdArray = new int[novelIds.Count];

        // Extract numbers from JSON array.
        string noveltyIds = "";
        for (int i = 0; i < novelIds.Count; i++) {
            novelIdArray[i] = novelIds[i].AsInt;
            noveltyIds += novelIdArray[i];
            if(i < novelIds.Count-1){
                noveltyIds += " ";
            }
        }

        int noveltyLevel = data[2]["novelty_level"].AsInt;
        UnityEngine.Debug.Log("novelty level " + noveltyLevel);
        String description = data[2]["novelty_description"];
        UnityEngine.Debug.Log("novelty description " + description);
        UnityEngine.Debug.Log("novelty " + msg.data);


        //record
        //EvaluationHandler.Instance.RecordNoveltyLikelihood(noveltyLikelihood,nonNoveltyLikelihood,noveltyIds,noveltyLevel,description);
        
        id = data [0];
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        json = JsonUtility.ToJson(msg);

        message = "[" + id + "," + json + "]";
    
#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    IEnumerator ReportNoveltyDescription(JSONNode data)
    {
        yield return new WaitForEndOfFrame();

        string id = data [0];

		Message msg = new Message();
        msg.data = "1";
        msg.time = DateTime.Now.ToString();

        string json = JsonUtility.ToJson(msg);

        string message = "[" + id + "," + json + "]";

        if(!isPlaying&&(LoadLevelSchema.Instance.isCheckPoint||LoadLevelSchema.Instance.isNewTrial||
        LoadLevelSchema.Instance.resumeTraining||LoadLevelSchema.Instance.evaluationTerminating)){            
            id = data [0];

            msg = new Message();
            msg.data = "0";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";

        }
        else{
            string noveltyLikelihood = data[2]["novelty_description"];
            //TODO record

            id = data [0];
            msg = new Message();
            msg.data = "1";
            msg.time = DateTime.Now.ToString();

            json = JsonUtility.ToJson(msg);

            message = "[" + id + "," + json + "]";
        }
#if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    IEnumerator ReadyForNewSet(JSONNode data)
    {

        yield return new WaitForEndOfFrame();

        //do not update levels file in is isCheckpoint state right after isNewTrial 
        //as they are updated when isNewTrial is true
        if(!LoadLevelSchema.Instance.doNotUpdateLevels){
            if(!LoadLevelSchema.Instance.firstLoad){
                LoadLevelSchema.Instance.updateLevelSet();
            }
            else{
                LoadLevelSchema.Instance.firstLoad = false;
            }
        }
        else{
            LoadLevelSchema.Instance.doNotUpdateLevels = false;
        }

        LoadLevelSchema.Instance.isCheckPoint = false;
        LoadLevelSchema.Instance.resumeTraining = false;
        
        if(LoadLevelSchema.Instance.isNewTrial){
            LoadLevelSchema.Instance.isNewTrial = false;
            //send the actual level set game state again after agent is ready for a new trial
            LoadLevelSchema.Instance.isCheckPoint = true;
            LoadLevelSchema.Instance.doNotUpdateLevels = true;
        } 
        if(LoadLevelSchema.Instance.needResetTimer){
            LoadLevelSchema.Instance.overallStopwatch.Reset();
            LoadLevelSchema.Instance.overallStopwatch.Start();
            LoadLevelSchema.Instance.needResetTimer = false;
        }

        if(LoadLevelSchema.Instance.resumeTraining){
            LevelList.Instance.CurrentIndex = LoadLevelSchema.Instance.prevLevelIndex;
        }
        else{
            LevelList.Instance.CurrentIndex = -1;
        }
//        LoadLevelSchema.Instance.overallStopwatch.Start();
        LoadLevelSchema.Instance.checkpointWatch.Reset();
        LoadLevelSchema.Instance.checkpointWatch.Start();
//        LevelLoader.LoadXmlLevel(LoadLevelSchema.Instance.currentXmlFiles[0]);
//        ABSceneManager.Instance.LoadScene("MainMenu");
        
        string id = data[0];
        string timeLimit,timeLeft,interactionLimit,interactionLeft,mode,seqOrSet,nLevels, attemptPerLevel, noveltyInfo;
        timeLimit = LoadLevelSchema.Instance.timeLimit.ToString();
        timeLeft = LoadLevelSchema.Instance.timeLeft.ToString();
        interactionLimit = LoadLevelSchema.Instance.interactionLimit.ToString();
        interactionLeft = LoadLevelSchema.Instance.interactionLeft.ToString();
        if(LoadLevelSchema.Instance.isTesting){
            mode = "testing";
        }
        else{
            mode = "training";
        }
        if(LoadLevelSchema.Instance.isSequence){
            seqOrSet = "seq";
        }
        else{
            seqOrSet = "set";
        }

        nLevels = LoadLevelSchema.Instance.currentXmlFiles.Length.ToString();
        attemptPerLevel = LoadLevelSchema.Instance.attemptsLimitPerLevel.ToString();
        TrialInfo trial = LoadLevelSchema.Instance.trials[LoadLevelSchema.Instance.currentTrialIndex];
        if(trial.notifyNovelty){
            noveltyInfo = "1";
        } 
        else{
            noveltyInfo = "0";
        }
        
        ABSceneManager.Instance.LoadScene("MainMenu");

        Message msg = new Message();
        msg.data = timeLimit + "," + timeLeft + "," + interactionLimit + "," + interactionLeft + "," + mode+","+seqOrSet +","+nLevels+","+attemptPerLevel+","+ noveltyInfo;
        msg.time = DateTime.Now.ToString();

        UnityEngine.Debug.Log("ready for new set" + msg.data);

        string json = JsonUtility.ToJson(msg);
        string message = "[" + id + "," + json + "]";
        #if UNITY_WEBGL && !UNITY_EDITOR
		socket.Send(System.Text.Encoding.UTF8.GetBytes(message));
#else
        socket.Send(message);
#endif
    }

    
    public void InitHandlers()
    {

        handlers = new Dictionary<string, Handler>();

        handlers["ReadyForNewSet"] = ReadyForNewSet;
        handlers["ReportNoveltyDescription"] = ReportNoveltyDescription;
        handlers["ReportNoveltyLikelihood"] = ReportNoveltyLikelihood;
        handlers["noveltyInfo"] = NoveltyInfo;
        handlers["click"] = Click;
        handlers["drag"] = Drag;
        handlers["tapShoot"] = TapShoot;
        handlers["shootAndRecordGroundTruth"] = ShootAndRecordGroundTruth;
        handlers["mousewheel"] = MouseWheel;
        handlers["screenshot"] = Screenshot;
        handlers["gamestate"] = GameState;
        handlers["loadscene"] = LoadScene;
        handlers["selectlevel"] = SelectLevel;
        handlers["selectNextAvailableLevel"] = SelectNextAvailableLevel;
        handlers["score"] = Score;
        handlers["NoOfLevels"] = NoOfLevels;
        handlers["NoisyGroundTruthWithScreenshot"] = NoisyGroundTruthWithScreenshot;
        handlers["NoisyGroundTruthWithoutScreenshot"] = NoisyGroundTruthWithoutScreenshot;
        
        if(LoadLevelSchema.Instance.devMode){//dev mode allows to use non-noisy ground truth (gt)
            handlers["GroundTruthWithScreenshot"] = GroundTruthWithScreenshot;
            handlers["GroundTruthWithoutScreenshot"] = GroundTruthWithoutScreenshot;
        }
        else{//in normal mode the non-noisy gt will be redirected to the noisy gt
            handlers["GroundTruthWithScreenshot"] = NoisyGroundTruthWithScreenshot;
            handlers["GroundTruthWithoutScreenshot"] = NoisyGroundTruthWithoutScreenshot;
        }

        handlers["SpeedUpGameSimulation"] = SpeedUpGameSimulation;
    }

    // Use this for initialization
    IEnumerator Start()
    {

        DontDestroyOnLoad(this.gameObject);
        isPlaying = false;
        
        this.isZoomingCompleted = true;// init to true and set to false right after loading a level

        //start connection after initialization of the game
        while(!LoadLevelSchema.Instance.initializationDone){
            yield return new WaitForEndOfFrame();            
        }
        InitHandlers();
        logPath = EvaluationHandler.savePath+"/agent.log";
        
        // read port from command line arguments, by default 9000 is used
        port = 9000;
        var args = System.Environment.GetCommandLineArgs();

        for (int i = 0; i < args.Length; i++) {
            UnityEngine.Debug.Log ("ARG " + i + ": " + args [i]);
            if (args [i] == "--port") {
                port = Int32.Parse(args [i + 1]);
                UnityEngine.Debug.Log("Assigned custom port: " + port);
            }
        }




        socket = new WebSocket(new Uri("ws://localhost:" + port + "/"));
        yield return StartCoroutine(socket.Connect());

        while (true)
        {

            string reply = socket.RecvString();

            if (reply != null)
            {

                JSONNode data = JSON.Parse(reply);

                string type = data[1];

                UnityEngine.Debug.Log("Received message: " + type);

                if (handlers[type] != null)
                {

                    StartCoroutine(handlers[type](data));
                }
                else
                {

                    UnityEngine.Debug.Log("Invalid message: " + type);
                }
            }

            if (socket.error != null)
            {

               //UnityEngine.Debug.Log("Error: " + socket.error);

                yield return new WaitForSeconds(1);

                socket = new WebSocket(new Uri("ws://localhost:" + port + "/"));
                yield return StartCoroutine(socket.Connect());
            }

            yield return 0;
        }

        //		socket.Close();
    }

    private IEnumerator GTTest(bool useNoise)
    {
        yield return new WaitForEndOfFrame();
        string savePath = "./gt/";
        Stopwatch stopwatch_overall = new Stopwatch();
        stopwatch_overall.Reset();
        stopwatch_overall.Start();
        SymbolicGameState gt = new SymbolicGameState(useNoise);
        string screenimage = gt.GetScreenshotStr();
        string gtdebug = gt.GetGTJson();
        UnityEngine.Debug.Log("total: " + stopwatch_overall.ElapsedTicks + " ms: " + stopwatch_overall.ElapsedMilliseconds);
        System.IO.File.WriteAllText(savePath + "GTData.json", gtdebug);
    }

    void Update(){

        /// For testing ground truth
        if (Input.GetKeyDown(KeyCode.Return))
        {
            bool useNoise = true;
            StartCoroutine(GTTest(useNoise));
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            bool useNoise = false;
            StartCoroutine(GTTest(useNoise));
        }
        /// End testing 

        string currentScence = SceneManager.GetActiveScene().name;

        if (currentScence == "GameWorld")
        {

            if (ABGameWorld.Instance.LevelCleared())
            {

                currentScence = "LevelCleared";
            }
            else if (ABGameWorld.Instance.LevelFailed())
            {

                currentScence = "LevelFailed";
            }
            else if (!ABGameWorld.Instance.IsLevelStable())
            {

                currentScence = "LevelUnstable";
            }
        }

        if(currentScence != "LevelUnstable" && currentScence != "GameWorld"){
            isPlaying = false;
        }
        else{
            isPlaying = true;
        }
    }

}
