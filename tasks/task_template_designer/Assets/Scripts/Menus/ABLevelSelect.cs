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
using System.IO;
using System.Collections;
using System.Linq;

public class ABLevelSelect : ABMenu {

	public int _lines = 5;

	public GameObject _levelSelector;
	public GameObject _canvas;

	public Vector2 _startPos;
	public Vector2 _buttonSize;

	private int _clickedButton;

	public Button goToLvevelBtn;
	public InputField levelNumberField;
	public Text levelRangeText;
	private string levelNumberStr;
	private int levelNumberInt;

	// Use this for initialization
	void Start()
	{
        // for physics test, load the levels in a constant order instead of using LoadLevelSchema
        string[] allXmlFiles = ABLevelUpdate.getAllXmlFiles();
        // string[] allXmlFiles = LoadLevelSchema.Instance.currentXmlFiles;
		_startPos.x = Mathf.Clamp(_startPos.x, 0, 1f) * Screen.width;
		_startPos.y = Mathf.Clamp(_startPos.y, 0, 1f) * Screen.height;

		LevelList.Instance.LoadLevelsFromSource(allXmlFiles);

		// Level selection through user input
		int totallevels = allXmlFiles.Length;
		levelRangeText = GameObject.Find("LevelRange").GetComponent<Text>();
		levelRangeText.text = "1 - " + totallevels;

		goToLvevelBtn = GameObject.Find("ButtonLoadLevel").GetComponent<Button>();
		levelNumberField = GameObject.Find("InputLevelNumber").GetComponent<InputField>();

		goToLvevelBtn.onClick.AddListener(delegate {
			levelNumberStr = levelNumberField.text;

			if (int.TryParse(levelNumberStr, out levelNumberInt))
			{
				if (levelNumberInt < 1 || levelNumberInt > totallevels)
				{
					Debug.Log("Wrong Level Number");
					ABSceneManager.Instance.LoadScene("LevelSelectMenu");
				}
				else
				{
					LevelList.Instance.SetLevel(levelNumberInt - 1);
					ABSceneManager.Instance.LoadScene("GameWorld");
				}
			}
			else {
				Debug.Log("Wrong Level Number");
				ABSceneManager.Instance.LoadScene("LevelSelectMenu");
			}

		});
	

	// read the value of the input field
	//InputField txt_Input = GameObject.Find("ObjectName").GetComponent<InputField>();

	//string ObjectsText = txt_Input.text;

	/*
	int j = 0;

	 for(int i = 0; i < allXmlFiles.Length; i++) {

		Vector2 pos = _startPos + new Vector2 ((i % _lines) * _buttonSize.x, j * _buttonSize.y);

		GameObject obj = Instantiate (_levelSelector, pos, Quaternion.identity) as GameObject;
		obj.transform.SetParent(_canvas.transform);

		ABLevelSelector sel = obj.AddComponent<ABLevelSelector> ();
		sel.LevelIndex = i;

		Button selectButton = obj.GetComponent<Button> ();

		selectButton.onClick.AddListener (delegate { 
			LoadNextScene("GameWorld", true, sel.UpdateLevelList); });

		Text selectText = selectButton.GetComponentInChildren<Text> ();
		selectText.text = "" + (i + 1);

		if ((i + 1) % _lines == 0)
			j--;
	} 
}*/
}

void Update()
{
	if (Input.GetKeyDown(KeyCode.Return))
	{
		goToLvevelBtn.onClick.Invoke();
	}
}


}
