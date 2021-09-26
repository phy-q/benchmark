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

﻿using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Threading;
using System.Diagnostics;
using UnityEngine.SceneManagement;

public class HUD : ABSingleton<HUD> {
	
	public float _zoomSpeed;
	public float _dragSpeed;

	public RectTransform  _scoreDisplay;

	public float _simulateDragTime = 1f;

	public float SimulatedTapTime = 1.0f;
	public int SimulateInputEvent { get; set; }

	public bool shootDone {get; set;}

	public Vector3 SimulateInputPos{ get; set; }
	public Vector3 SimulateInputDelta{ get; set; }

	private bool _isZoomingIn; 
	private bool _isZoomingOut;
	private bool _usedSpecialPower;

	private int _totalScore;
	private float _simulatedDragTimer;

	private Vector3 _inputPos;
	private Vector3 _dragOrigin;
	private ABBird _selectedBird;

	private long startTime;

	private Stopwatch stopwatch;
	void Start() {
		stopwatch = new Stopwatch();
		SetScoreDisplay(_totalScore);
	}

	// Update is called once per frame
	void Update () {

		float scrollDirection = Input.GetAxis("Mouse ScrollWheel");

		if(scrollDirection != 0f) {

			if (scrollDirection > 0f)
				_isZoomingIn = true;
			
			else if (scrollDirection < 0f)
				_isZoomingOut = true;
		}
		
		if(_isZoomingIn) {
			
			if (scrollDirection != 0f) {

				// Zoom triggered via MouseWheel
				_isZoomingIn = false;
				CameraZoom(-ABConstants.MOUSE_SENSIBILITY);
			} 
			else {

				// Zoom triggered via HUD
				CameraZoom(-1f);
			}
			
			return;
		}
		
		if(_isZoomingOut) {
			
			if (scrollDirection != 0f) {

				// Zoom triggered via MouseWheel
				_isZoomingOut = false;
				CameraZoom (ABConstants.MOUSE_SENSIBILITY);
			} 
			else {

				// Zoom triggered via HUD
				CameraZoom(1f);
			}

			return;
		}

		bool isMouseControlling = true;
		_inputPos = Input.mousePosition;

		if (SimulateInputEvent > 0) {
			_inputPos = SimulateInputPos;

			//disable mouse interaction when the agent is playing
			isMouseControlling = false;
		}

		if(Input.GetMouseButtonDown(0) || SimulateInputEvent == 1) {

			ClickDown (_inputPos);

			if(!isMouseControlling)
				SimulateInputEvent++;
			
		}
		else if(Input.GetMouseButton(0) || SimulateInputEvent == 2) {

			if(SimulateInputEvent > 0 && !isMouseControlling)
				
				Drag (SimulateInputDelta);
			else
				Drag (_inputPos);

			_simulatedDragTimer += Time.deltaTime;
			if (_simulatedDragTimer >= _simulateDragTime) {

				if(!isMouseControlling)
					SimulateInputEvent++;
				
				_simulatedDragTimer = 0f;
			}
        }
		else if(Input.GetMouseButtonUp(0) || SimulateInputEvent == 3) {
			ClickUp ();
			
			stopwatch.Reset();
        	stopwatch.Start();
			if(!isMouseControlling) {
				startTime = stopwatch.ElapsedMilliseconds;
				SimulateInputEvent++;
			}
			
		} 
		else if(Input.GetMouseButtonUp(0) || SimulateInputEvent >= 4) {
			// Wait for tap time 
//			UnityEngine.Debug.Log("sim step 4");
			
			long tick =  stopwatch.ElapsedTicks;
			long elapsedTime = stopwatch.ElapsedMilliseconds;
//			UnityEngine.Debug.Log("start time " + startTime);
//			UnityEngine.Debug.Log("elapsedTime " + elapsedTime);	
			
			//do not tap if tap time <= 0
			if(SimulatedTapTime <= 0){
				SimulateInputEvent = 0;
				shootDone = true;			
			}		
			else if (elapsedTime >= SimulatedTapTime) {
//				UnityEngine.Debug.Log("in tap time block");
				if (_selectedBird && _selectedBird.IsInFrontOfSlingshot () &&	
		    	_selectedBird == ABGameWorld.Instance.GetCurrentBird () && 
				!_selectedBird.IsDying && !_usedSpecialPower) {
					_usedSpecialPower = true;
					_selectedBird.SendMessage ("SpecialAttack", SendMessageOptions.DontRequireReceiver);

				}

				SimulateInputEvent = 0;
				shootDone = true;			
//				UnityEngine.Debug.Log("in tap time shootDone is " + shootDone);
			}

		}

		string currentScence = SceneManager.GetActiveScene().name;

		if (currentScence == "GameWorld") {

			if (ABGameWorld.Instance.LevelCleared () || ABGameWorld.Instance.LevelFailed ()) {
				SimulateInputEvent = 0;
				shootDone = true;
			}
			
		}
		
	}

	void LateUpdate() {
		
		if(Input.GetMouseButton(0))
			_dragOrigin = _inputPos;
		
	}

	private void ClickDown(Vector3 position) {

		_dragOrigin = position;
		
		Ray ray = Camera.main.ScreenPointToRay(position);
		RaycastHit2D hit = Physics2D.Raycast(ray.origin, ray.direction);

		if (hit && hit.transform.tag == "Bird") {
			//UnityEngine.Debug.Log("Bird touched to release!");
			_selectedBird = hit.transform.gameObject.GetComponent<ABBird> ();
			if (_selectedBird && !_selectedBird.IsSelected && 
				_selectedBird == ABGameWorld.Instance.GetCurrentBird ()) {
				
				_selectedBird.SelectBird ();
				_usedSpecialPower = false;
				return;
			}
		} 
			
		// Trigger special attack
		if (_selectedBird && _selectedBird.IsInFrontOfSlingshot () &&
		    _selectedBird == ABGameWorld.Instance.GetCurrentBird () && 
			!_selectedBird.IsDying && !_usedSpecialPower) {
			_usedSpecialPower = true;
			_selectedBird.SendMessage ("SpecialAttack", SendMessageOptions.DontRequireReceiver);
		}
	}

	private void ClickUp() {

		if (_selectedBird) {
			
			if (!_selectedBird.IsFlying && !_selectedBird.IsDying && 
				_selectedBird == ABGameWorld.Instance.GetCurrentBird ()) {

				_selectedBird.LaunchBird ();
				//UnityEngine.Debug.Log("Bird Released!");
				EvaluationHandler.Instance.RecordEvaluationScore("Bird Release");
			}
		}
	}

	private void Drag(Vector3 position) {

		if(_selectedBird) {

			if (!_selectedBird.IsFlying && !_selectedBird.IsDying && 
				_selectedBird == ABGameWorld.Instance.GetCurrentBird ()) {
				
				Vector3 dragPosition = Camera.main.ScreenToWorldPoint(position);
				dragPosition = new Vector3(dragPosition.x, dragPosition.y, _selectedBird.transform.position.z);

				_selectedBird.DragBird(dragPosition);
			}

		}
		else {
			
			Vector3 dragPosition = position - _dragOrigin;
			ABGameWorld.Instance.GameplayCam.DragCamera(dragPosition * _dragSpeed * Time.fixedDeltaTime);
		}
	}

	private void SetZoomIn(bool zoomIn) {
		
		_isZoomingIn = zoomIn;
	}
	
	private void SetZoomOut(bool zoomOut) {
		
		_isZoomingOut = zoomOut;
	}
	
	public void CameraZoom(float scrollDirection) {

		ABGameWorld.Instance.GameplayCam.ZoomCamera(scrollDirection * _zoomSpeed * Time.deltaTime);
	}

	private void SetScoreDisplay(int score) {
		
		if(_scoreDisplay) {
			
			_totalScore = score;
			_scoreDisplay.GetComponent<Text>().text = _totalScore.ToString();
		}
	}

	public void AddScore(int score) {
		
		_totalScore += score;

		// prevent the total score going negative
		if (_totalScore < 0)
			_totalScore = 0;

		_scoreDisplay.GetComponent<Text>().text = _totalScore.ToString();
	}

	public int GetScore() {

		return _totalScore;
	}
}
