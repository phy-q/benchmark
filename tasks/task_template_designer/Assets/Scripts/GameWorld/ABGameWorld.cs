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
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System.Collections;
using System.Collections.Generic;
using System.Xml;
using System.IO;
using UnityEditor;
using System.Text.RegularExpressions;

public class ABGameWorld : ABSingleton<ABGameWorld>
{

    static int _levelTimesTried;
    static public float SimulationSpeed = 1.0f;

    // novelty type for the novelty level 3 (default -1 means no level 3 novelty is used)
    static public int noveltyTypeForNovelty3 = -1;

    // when set to other than -1, ground truth will be taken every n frames
    static public int takeGroundTruthEveryNthFrames = 5;
    // when set to true, we will record ground truth every n frames
    static public bool wasBirdLaunched = false;

    static public bool isRecordingBatchGroundTruth = false;

    static public bool isBannerShowing = false;

    static public List<(string, string)> batchGroundTruths = new List<(string, string)>();
    static public List<string> batchScreenshots = new List<string>();

    private bool _levelCleared;

    public List<ABPig> _pigs;
    public List<ABBird> _birds;
    private int framesPassedSinceLastTake = 1;
    private List<ABParticle> _birdTrajectory;

    private ABBird _lastThrownBird;
    public Transform blocksTransform{get; private set;}
    public Transform birdsTransform{get; private set;}
    public Transform plaftformsTransform{get; private set;}
    public Transform slingshotBaseTransform{get; private set;}

    private GameObject _slingshot;
    public GameObject Slingshot() { return _slingshot; }

    private GameObject _levelFailedBanner;
    public bool LevelFailed() { 
        if(_levelFailedBanner==null){
            return false;
        }
        return _levelFailedBanner.activeSelf; 
    }

    private GameObject _levelClearedBanner;
    public bool LevelCleared() {
        if(_levelClearedBanner==null){
            return false;
        } 
        return _levelClearedBanner.activeSelf; 
    }

    private int _pigsAtStart;
    public int PigsAtStart { get { return _pigsAtStart; } }

    private int _birdsAtStart;
    public int BirdsAtStart { get { return _birdsAtStart; } }

    private int _blocksAtStart;
    public int BlocksAtStart { get { return _blocksAtStart; } }

    public ABGameplayCamera GameplayCam { get; set; }
    public float LevelWidth { get; set; }
    public float LevelHeight { get; set; }

    private bool BirdsScoreUpdated = false;

    //Score get set
    public float LevelScore { get; set; }

    // Game world properties
    public bool _isSimulation;
    public int _timesToGiveUp;
    public float _timeToResetLevel = 1f;
    public int _birdsAmounInARow = 5;

    public AudioClip[] _clips;

    public static AssetBundle NOVELTIES;

    void Awake()
    {
        isBannerShowing = false;
        blocksTransform = GameObject.Find("Blocks").transform;
        birdsTransform = GameObject.Find("Birds").transform;
        plaftformsTransform = GameObject.Find("Platforms").transform;

        _levelFailedBanner = GameObject.Find("LevelFailedBanner").gameObject;
        _levelFailedBanner.gameObject.SetActive(false);

        _levelClearedBanner = GameObject.Find("LevelClearedBanner").gameObject;
        _levelClearedBanner.gameObject.SetActive(false);

        GameplayCam = GameObject.Find("Camera").GetComponent<ABGameplayCamera>();
    }

    // Use this for initialization
    void Start()
    {
        isBannerShowing = false;
        _pigs = new List<ABPig>();
        _birds = new List<ABBird>();
        _birdTrajectory = new List<ABParticle>();

        _levelCleared = false;

        if (!_isSimulation)
        {

            GetComponent<AudioSource>().PlayOneShot(_clips[0]);
            GetComponent<AudioSource>().PlayOneShot(_clips[1]);
        }

        // If there are objects in the scene, use them to play
        if (blocksTransform.childCount > 0 || birdsTransform.childCount > 0)
        {

            foreach (Transform bird in birdsTransform)
                AddBird(bird.GetComponent<ABBird>());

            foreach (Transform block in blocksTransform)
            {
                ABPig pig = block.GetComponent<ABPig>();
                if (pig != null){
                    _pigs.Add(pig);
                }

            }

        }
        else
        {
            //UnityEngine.Debug.Log("Game level loaded!");
            ABLevel currentLevel = LevelList.Instance.GetCurrentLevel();
            
            if (currentLevel != null)
            {
                // check whether the novelty type 3 is used

                CleanCache();
                if (NOVELTIES != null)
                {

                    NOVELTIES.Unload(true);
                    //Debug.Log("asset bundle unloaded");
                }

                NOVELTIES = AssetBundle.LoadFromFile(currentLevel.assetBundleFilePath);
                ClearWorld();

                try
                {
                    TextAsset configFile = NOVELTIES.LoadAsset("level_3_config.txt") as TextAsset;
                    string[] configFileData = configFile.text.Split('\n');

                    // extract the novelty type from the file
                    string noveltyTypeString = Regex.Match(configFileData[0], @"\d+").Value;
                    noveltyTypeForNovelty3 = int.Parse(noveltyTypeString);

                    //Debug.Log("Level 3 Novelty Type From Asset Bundle: " + noveltyTypeForNovelty3);


                }
                catch (System.Exception)
                {
                    Debug.Log("No any evidence for novelty type 3 in the asset bundle");
                    noveltyTypeForNovelty3 = -1;
                }

                DecodeLevel(currentLevel);
                AdaptCameraWidthToLevel();
                //UnityEngine.Debug.Log("Game level loaded!");
                EvaluationHandler.Instance.RecordEvaluationScore("Level Loaded");
                _levelTimesTried = 0;

                slingshotBaseTransform = GameObject.Find("slingshot_base").transform;
            }
        }
    }

    public void DecodeLevel(ABLevel currentLevel)
    {
        isBannerShowing = false;
        CleanCache();
        if (NOVELTIES != null) {
            
            NOVELTIES.Unload(true);
            Debug.Log("asset bundle unloaded");
        }
        
	    NOVELTIES = AssetBundle.LoadFromFile(currentLevel.assetBundleFilePath);
        ClearWorld();


        LevelHeight = ABConstants.LEVEL_ORIGINAL_SIZE.y;
        LevelWidth = (float)currentLevel.width * ABConstants.LEVEL_ORIGINAL_SIZE.x;

        Vector3 cameraPos = GameplayCam.transform.position;
        cameraPos.x = currentLevel.camera.x;
        cameraPos.y = currentLevel.camera.y;
        GameplayCam.transform.position = cameraPos;

        GameplayCam._minWidth = currentLevel.camera.minWidth;
        GameplayCam._maxWidth = currentLevel.camera.maxWidth;

        Vector3 landscapePos = ABWorldAssets.LANDSCAPE.transform.position;
        Vector3 backgroundPos = ABWorldAssets.BACKGROUND.transform.position;

        if (currentLevel.width > 1)
        {

            landscapePos.x -= LevelWidth / 4f;
            backgroundPos.x -= LevelWidth / 4f;
        }

        for (int i = 0; i < currentLevel.width; i++)
        {

            GameObject landscape = (GameObject)Instantiate(ABWorldAssets.LANDSCAPE, landscapePos, Quaternion.identity);
            landscape.transform.parent = transform;

            float screenRate = currentLevel.camera.maxWidth / LevelHeight;
            if (screenRate > 2f)
            {

                for (int j = 0; j < (int)screenRate; j++)
                {

                    Vector3 deltaPos = Vector3.down * (LevelHeight / 1.5f + (j * 2f));
                    Instantiate(ABWorldAssets.GROUND_EXTENSION, landscapePos + deltaPos, Quaternion.identity);
                }
            }

            landscapePos.x += ABConstants.LEVEL_ORIGINAL_SIZE.x - 0.01f;

            GameObject background = (GameObject)Instantiate(ABWorldAssets.BACKGROUND, backgroundPos, Quaternion.identity);
            background.transform.parent = GameplayCam.transform;
            backgroundPos.x += ABConstants.LEVEL_ORIGINAL_SIZE.x - 0.01f;
        }

        //Reading the score
        LevelScore = currentLevel.score.highScore;

        Vector2 slingshotPos = new Vector2(currentLevel.slingshot.x, currentLevel.slingshot.y);
        _slingshot = (GameObject)Instantiate(ABWorldAssets.SLINGSHOT, slingshotPos, Quaternion.identity);
        _slingshot.name = "Slingshot";
        _slingshot.transform.parent = transform;

        foreach (BirdData gameObj in currentLevel.birds)
        {

            if (!gameObj.type.Contains("novel"))
            {
                AddBird(ABWorldAssets.BIRDS[gameObj.type], ABWorldAssets.BIRDS[gameObj.type].transform.rotation);
            }

            else
            {
                GameObject newBird = (GameObject)NOVELTIES.LoadAsset(gameObj.type);
                string matrialName = "novel_material_" + gameObj.type.Split('_')[2];
                newBird.GetComponent<PolygonCollider2D>().sharedMaterial = (PhysicsMaterial2D)NOVELTIES.LoadAsset(matrialName);
                AddBird(newBird, newBird.transform.rotation);

            }

        }

        foreach (OBjData gameObj in currentLevel.pigs)
        {

            Vector2 pos = new Vector2(gameObj.x, gameObj.y);
            Quaternion rotation = Quaternion.Euler(0, 0, gameObj.rotation);


            if (!gameObj.type.Contains("novel"))
            {
                AddPig(ABWorldAssets.PIGS[gameObj.type], pos, rotation);
            }

            else
            {
                GameObject newPig = (GameObject)NOVELTIES.LoadAsset(gameObj.type);
                string matrialName = "novel_material_" + gameObj.type.Split('_')[2];
                newPig.GetComponent<PolygonCollider2D>().sharedMaterial = (PhysicsMaterial2D)NOVELTIES.LoadAsset(matrialName);
                AddPig(newPig, pos, rotation);

            }

        }

        foreach (BlockData gameObj in currentLevel.blocks)
        {

            Vector2 pos = new Vector2(gameObj.x, gameObj.y);
            Quaternion rotation = Quaternion.Euler(0, 0, gameObj.rotation);

            if (!gameObj.type.Contains("novel"))
            {
                GameObject block = AddBlock(ABWorldAssets.BLOCKS[gameObj.type], pos, rotation);

                MATERIALS material = (MATERIALS)System.Enum.Parse(typeof(MATERIALS), gameObj.material);

                block.GetComponent<ABBlock>().SetMaterial(material);

            }

            else
            {
                GameObject newBlock = (GameObject)NOVELTIES.LoadAsset(gameObj.type);
                GameObject block = AddBlock(newBlock, pos, rotation);

                string matrialName = "novel_material_" + gameObj.type.Split('_')[2];

                block.GetComponent<ABBlock>().SetMaterial(MATERIALS.novelty, matrialName);
            }


        }

        foreach (PlatData gameObj in currentLevel.platforms)
        {

            Vector2 pos = new Vector2(gameObj.x, gameObj.y);
            Quaternion rotation = Quaternion.Euler(0, 0, gameObj.rotation);

            AddPlatform(ABWorldAssets.PLATFORM, pos, rotation, gameObj.scaleX, gameObj.scaleY);
        }

        foreach (OBjData gameObj in currentLevel.tnts)
        {

            Vector2 pos = new Vector2(gameObj.x, gameObj.y);
            Quaternion rotation = Quaternion.Euler(0, 0, gameObj.rotation);

            AddBlock(ABWorldAssets.TNT, pos, rotation);
        }


        StartWorld();
    }

    // Update is called once per frame
    void Update()
    {
        // Check if birds was trown, if it died and swap them when needed
        ManageBirds();

        if(wasBirdLaunched == true && isRecordingBatchGroundTruth == true) {
            if(framesPassedSinceLastTake >= takeGroundTruthEveryNthFrames) {
                // Every n-frames take ground truth once the shot is made
                Debug.Log("recording a frame");
                bool useNoise = LoadLevelSchema.Instance.devMode;
                SymbolicGameState gt = new SymbolicGameState(useNoise);

                string gtjson = gt.GetGTJson();
                string image = gt.GetScreenshotStr();

                batchGroundTruths.Add((image, gtjson));

                framesPassedSinceLastTake = 1; // 1 since every n-th frame
            }
            else {
                framesPassedSinceLastTake += 1;
            }
        }

        // Check if the level is stable and stop recording the ground truth if needed ### commented by Chathura to avoid issues loading levels with level editor ###
        /*if(IsLevelStable() == true && wasBirdLaunched == true) {
            wasBirdLaunched = false;
            framesPassedSinceLastTake = 1; // reset
            isRecordingBatchGroundTruth = false;
            UnityEngine.Debug.Log("Done recording in ABGameWorld");
        }*/

        // Speed up or slow down the game on the request
        if (Time.timeScale != SimulationSpeed)
        {
            UnityEngine.Debug.Log("Setting sim speed game");
            Time.timeScale = SimulationSpeed;
        }

    }

    public bool IsObjectOutOfWorld(Transform abGameObject, Collider2D abCollider)
    {

        Vector2 halfSize = abCollider.bounds.size / 2f;

        if (abGameObject.position.x - halfSize.x > LevelWidth / 2f ||
           abGameObject.position.x + halfSize.x < -LevelWidth / 2f)

            return true;

        return false;
    }

    void ManageBirds()
    {

        if (_birds.Count == 0)
            return;

        // Move next bird to the slingshot
        if (_birds[0].JumpToSlingshot)
            _birds[0].SetBirdOnSlingshot();

        //		int birdsLayer = LayerMask.NameToLayer("Birds");
        //		int blocksLayer = LayerMask.NameToLayer("Blocks");
        //		if(_birds[0].IsFlying || _birds[0].IsDying)
        //			
        //			Physics2D.IgnoreLayerCollision(birdsLayer, blocksLayer, false);
        //		else 
        //			Physics2D.IgnoreLayerCollision(birdsLayer, blocksLayer, true);
    }

    public ABBird GetCurrentBird()
    {

        if (_birds.Count > 0)
            return _birds[0];

        return null;
    }

    public void NextLevel()
    {

        // update the level list when the level cleared
        // not used in test harness
//        ABLevelUpdate.RefreshLevelList();
        isBannerShowing = false;
        if (LevelList.Instance.NextLevel() == null)
            ABSceneManager.Instance.LoadScene("MainMenu");
        else
            ABSceneManager.Instance.LoadScene(SceneManager.GetActiveScene().name);
    }

    public void ResetLevel()
    {

        isBannerShowing = false;
        if (_levelFailedBanner.activeSelf)
            _levelTimesTried++;

        ABSceneManager.Instance.LoadScene(SceneManager.GetActiveScene().name);
        //UnityEngine.Debug.Log("level reloaded!");
        EvaluationHandler.Instance.RecordEvaluationScore("Level reload");
    }

    public void AddTrajectoryParticle(ABParticle trajectoryParticle)
    {

        _birdTrajectory.Add(trajectoryParticle);
    }

    public void RemoveLastTrajectoryParticle()
    {

        foreach (ABParticle part in _birdTrajectory)
            part.Kill();
    }

    public void AddBird(ABBird readyBird)
    {

        if (_birds.Count == 0)
            readyBird.GetComponent<Rigidbody2D>().gravityScale = 0f;

        if (readyBird != null)
            _birds.Add(readyBird);
    }

    public GameObject AddBird(GameObject original, Quaternion rotation)
    {

        Vector3 birdsPos = _slingshot.transform.position - ABConstants.SLING_SELECT_POS;

        if (_birds.Count >= 1)
        {

            birdsPos.y = _slingshot.transform.position.y;

            for (int i = 0; i < _birds.Count; i++)
            {

                if ((i + 1) % _birdsAmounInARow == 0)
                {

                    float coin = (Random.value < 0.5f ? 1f : -1);
                    birdsPos.x = _slingshot.transform.position.x + (Random.value * 0.5f * coin);
                }

                birdsPos.x -= original.GetComponent<SpriteRenderer>().bounds.size.x * 1.75f;
            }
        }

        GameObject newGameObject = (GameObject)Instantiate(original, birdsPos, rotation);
        Vector3 scale = newGameObject.transform.localScale;
        scale.x = original.transform.localScale.x;
        scale.y = original.transform.localScale.y;
        newGameObject.transform.localScale = scale;

        newGameObject.transform.parent = birdsTransform;
        newGameObject.name = "bird_" + _birds.Count;

        ABBird bird = newGameObject.GetComponent<ABBird>();
        bird.SendMessage("InitSpecialPower", SendMessageOptions.DontRequireReceiver);

        if (_birds.Count == 0)
            bird.GetComponent<Rigidbody2D>().gravityScale = 0f;

        if (bird != null)
            _birds.Add(bird);

        return newGameObject;
    }

    public GameObject AddPig(GameObject original, Vector3 position, Quaternion rotation, float scale = 1f)
    {

        GameObject newGameObject = AddBlock(original, position, rotation, scale);

        ABPig pig = newGameObject.GetComponent<ABPig>();
        if (pig != null)
            _pigs.Add(pig);

        return newGameObject;
    }

    public GameObject AddPlatform(GameObject original, Vector3 position, Quaternion rotation, float scaleX = 1f, float scaleY = 1f)
    {

        GameObject platform = AddBlock(original, position, rotation, scaleX, scaleY);
        platform.transform.parent = plaftformsTransform;

        return platform;
    }

    public GameObject AddBlock(GameObject original, Vector3 position, Quaternion rotation, float scaleX = 1f, float scaleY = 1f)
    {

        GameObject newGameObject = (GameObject)Instantiate(original, position, rotation);
        newGameObject.transform.parent = blocksTransform;

        Vector3 newScale = newGameObject.transform.localScale;
        newScale.x = scaleX;
        newScale.y = scaleY;
        newGameObject.transform.localScale = newScale;

        return newGameObject;
    }

    private void ShowLevelFailedBanner()
    {
        //MaxScoreUpdate();
        if (_levelCleared)
            return;

        if (!IsLevelStable())
        {

            Invoke("ShowLevelFailedBanner", 1f);
        }
        else
        { // Player lost the game
            // avoid multiple invoking of the function adding multiple scores
            if (!BirdsScoreUpdated)
            {

                // add points to the remaining birds
                ScoreHud.Instance.SpawnScorePoint(10000 * (int)_birds.Count, transform.position);
                BirdsScoreUpdated = true;

                //For evaluation purpose
                EvaluationHandler.Instance.RecordEvaluationScore("Fail");
            }

            HUD.Instance.gameObject.SetActive(false);

            if (_levelTimesTried < _timesToGiveUp - 1)
            {
                _levelFailedBanner.SetActive(true);
                Text[] TextFieldsInBanner = _levelFailedBanner.GetComponentsInChildren<Text>();
                TextFieldsInBanner[0].text = "Level Failed!";
                TextFieldsInBanner[1].text = "Score: " + HUD.Instance.GetScore().ToString();

            }
            else
            {
                _levelClearedBanner.SetActive(true);
                Text[] TextFieldsInBanner = _levelClearedBanner.GetComponentsInChildren<Text>();
                TextFieldsInBanner[0].text = "Level Failed!";
                TextFieldsInBanner[1].text = "Score: " + HUD.Instance.GetScore().ToString();


            }
            isBannerShowing = true;
        }


    }

    private void ShowLevelClearedBanner()
    {
        if (!IsLevelStable())
        {
            Invoke("ShowLevelClearedBanner", 1f);
        }
        else
        { // Player won the game

            // avoid multiple invoking of the function adding multiple scores
            if (!BirdsScoreUpdated)
            {
                // add points to the remaining birds
                ScoreHud.Instance.SpawnScorePoint(10000 * (int)_birds.Count, transform.position);
                BirdsScoreUpdated = true;

                //For evaluation purpose
                EvaluationHandler.Instance.RecordEvaluationScore("Pass");
            }

            HUD.Instance.gameObject.SetActive(false);

            _levelClearedBanner.SetActive(true);
            Text[] TextFieldsInBanner = _levelClearedBanner.GetComponentsInChildren<Text>();
            TextFieldsInBanner[0].text = "Level Cleared!";
            TextFieldsInBanner[1].text = "Score: " + HUD.Instance.GetScore().ToString();
            //MaxScoreUpdate();
            isBannerShowing = true;
        }

    }

    public void KillPig(ABPig pig)
    {

        _pigs.Remove(pig);

        if (_pigs.Count == 0)
        {

            // ScoreHud.Instance.SpawnScorePoint(10000*(uint)_birds.Count, transform.position); 

            // Check if player won the game
            if (!_isSimulation)
            {
                _levelCleared = true;
                Invoke("ShowLevelClearedBanner", _timeToResetLevel);
            }

            return;
        }
    }

    public void nextbirdonsling()
    {
        /*
        if (!IsLevelStable())
        {
            Invoke("nextbirdonsling", 1f);
        }
        else
        {
            _birds[0].GetComponent<Rigidbody2D>().gravityScale = 0f;
            _birds[0].JumpToSlingshot = true;
        }*/
        _birds[0].GetComponent<Rigidbody2D>().gravityScale = 0f;
        _birds[0].JumpToSlingshot = true;

    }

    public void KillBird(ABBird bird)
    {
        if (!_birds.Contains(bird))
            return;

        _birds.Remove(bird);

        if (_birds.Count == 0)
        {
            // Check if player lost the game
            if (!_isSimulation) {
                Invoke("ShowLevelFailedBanner", _timeToResetLevel);

            }

            return;
        }
        nextbirdonsling();
    }

    public int GetPigsAvailableAmount()
    {

        return _pigs.Count;
    }

    public int GetBirdsAvailableAmount()
    {

        return _birds.Count;
    }

    public int GetBlocksAvailableAmount()
    {

        int blocksAmount = 0;

        foreach (Transform b in blocksTransform)
        {

            if (b.GetComponent<ABPig>() == null)

                for (int i = 0; i < b.GetComponentsInChildren<Rigidbody2D>().Length; i++)
                    blocksAmount++;
        }

        return blocksAmount;
    }

    public bool IsLevelStable()
    {
        return GetLevelStability() == 0f;
    }

    public float GetLevelStability()
    {

        float totalVelocity = 0f;

        foreach (Transform b in blocksTransform)
        {

            Rigidbody2D[] bodies = b.GetComponentsInChildren<Rigidbody2D>();

            foreach (Rigidbody2D body in bodies)
            {

                if (!IsObjectOutOfWorld(body.transform, body.GetComponent<Collider2D>()))
                    totalVelocity += body.velocity.magnitude;
            }
        }

        foreach (Transform b in birdsTransform)
        {

            Rigidbody2D[] bodies = b.GetComponentsInChildren<Rigidbody2D>();

            foreach (Rigidbody2D body in bodies)
            {

                if (!IsObjectOutOfWorld(body.transform, body.GetComponent<Collider2D>()))
                    totalVelocity += body.velocity.magnitude;
            }
        }

        return totalVelocity;
    }

    public List<GameObject> BlocksInScene()
    {

        List<GameObject> objsInScene = new List<GameObject>();

        foreach (Transform b in blocksTransform)
            objsInScene.Add(b.gameObject);

        return objsInScene;
    }

    public Vector3 DragDistance()
    {

        Vector3 selectPos = (_slingshot.transform.position - ABConstants.SLING_SELECT_POS);
        return slingshotBaseTransform.transform.position - selectPos;
    }

    public void SetSlingshotBaseActive(bool isActive)
    {

        slingshotBaseTransform.gameObject.SetActive(isActive);
    }

    public void ChangeSlingshotBasePosition(Vector3 position)
    {

        slingshotBaseTransform.transform.position = position;
    }

    public void ChangeSlingshotBaseRotation(Quaternion rotation)
    {

        slingshotBaseTransform.transform.rotation = rotation;
    }

    public bool IsSlingshotBaseActive()
    {

        return slingshotBaseTransform.gameObject.activeSelf;
    }

    public Vector3 GetSlingshotBasePosition()
    {

        return slingshotBaseTransform.transform.position;
    }

    public void StartWorld()
    {

        _pigsAtStart = GetPigsAvailableAmount();
        _birdsAtStart = GetBirdsAvailableAmount();
        _blocksAtStart = GetBlocksAvailableAmount();

        //For evaluation purposes
        //EvaluationHandler.Instance.RecordStartData();
    }

    public void ClearWorld()
    {

        foreach (Transform b in blocksTransform)
            Destroy(b.gameObject);

        _pigs.Clear();

        foreach (Transform b in birdsTransform)
            Destroy(b.gameObject);

        _birds.Clear();
    }

    private void AdaptCameraWidthToLevel()
    {

        Collider2D[] bodies = blocksTransform.GetComponentsInChildren<Collider2D>();

        if (bodies.Length == 0)
            return;

        // Adapt the camera to show all the blocks		
        float levelLeftBound = -LevelWidth / 2f;
        float groundSurfacePos = LevelHeight / 2f;

        float minPosX = Mathf.Infinity;
        float maxPosX = -Mathf.Infinity;
        float maxPosY = -Mathf.Infinity;

        // Get position of first non-empty stack
        for (int i = 0; i < bodies.Length; i++)
        {
            float minPosXCandidate = bodies[i].transform.position.x - bodies[i].bounds.size.x / 2f;
            if (minPosXCandidate < minPosX)
                minPosX = minPosXCandidate;

            float maxPosXCandidate = bodies[i].transform.position.x + bodies[i].bounds.size.x / 2f;
            if (maxPosXCandidate > maxPosX)
                maxPosX = maxPosXCandidate;

            float maxPosYCandidate = bodies[i].transform.position.y + bodies[i].bounds.size.y / 2f;
            if (maxPosYCandidate > maxPosY)
                maxPosY = maxPosYCandidate;
        }

        float cameraWidth = Mathf.Abs(minPosX - levelLeftBound) +
            Mathf.Max(Mathf.Abs(maxPosX - minPosX), Mathf.Abs(maxPosY - groundSurfacePos)) + 0.5f;

        GameplayCam.SetCameraWidth(cameraWidth);
    }

    public void MaxScoreUpdate()
    {
        float totalScore = HUD.Instance.GetScore();

        //read the highScore from Xml file
        float currentScore = LevelScore;

        int currentIndex = LevelList.Instance.CurrentIndex;
//        string[] levelFilePaths = ABLevelUpdate.levelPath.ToArray();
        string[] levelFilePaths = LoadLevelSchema.Instance.currentLevelPaths;
        if (totalScore > currentScore)
        {
            string xmldata = levelFilePaths[currentIndex];
            string levelText = LevelLoader.ReadXmlLevel(xmldata);
            string newText = "<Score highScore =\"" + totalScore.ToString() + "\">";
            string oldText = "<Score highScore =\"" + currentScore.ToString() + "\">";
            levelText = levelText.Replace(oldText, newText);

            File.WriteAllText(xmldata, levelText);
        }

    }

    public static void CleanCache()
    {
        if (Caching.ClearCache())
        {
            Debug.Log("Successfully cleaned the cache.");
        }
        else
        {
            Debug.Log("Cache is being used.");
        }
    }
}
