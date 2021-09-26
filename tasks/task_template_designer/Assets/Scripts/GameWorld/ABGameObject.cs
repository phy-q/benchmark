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
using System;
using System.Collections;

[RequireComponent(typeof(Collider2D))]
[RequireComponent(typeof(Rigidbody2D))]
[RequireComponent(typeof(SpriteRenderer))]
[RequireComponent(typeof(ABParticleSystem))]
public class ABGameObject : MonoBehaviour
{
    private float _currentLife = 10000;

    protected int _spriteChangedTimes;

    protected Collider2D _collider;
    protected Rigidbody2D _rigidBody;
    protected SpriteRenderer _spriteRenderer;
    protected ABParticleSystem _destroyEffect;

    public Sprite[] _sprites;

    public float _life = 10f;
    public float _timeToDie = 1f;

    /// <summary>
    /// Density of an object. The mass of rigidbody will be mulitiplied by this value in Start(). 
    /// Therefore, the value of mass set in inspector means the "size" of this object.
    /// </summary>
    public float _density = 1f;

    /// <summary>
    /// Defense of an object
    /// </summary>
    public float _defense;

    // Velocity of the rigidbody of this object in last frame
    public Vector3 lastVelocity;

    public bool IsDying { get; set; }

    protected virtual void Awake()
    {

        _collider = GetComponent<Collider2D>();
        _rigidBody = GetComponent<Rigidbody2D>();
        _destroyEffect = GetComponent<ABParticleSystem>();
        _spriteRenderer = GetComponent<SpriteRenderer>();

        IsDying = false;
    }

    protected virtual void Start()
    {
        // mass = area * density
        _collider.density = _density;
        _rigidBody.useAutoMass = true;

        // The object is immortal in the first n secs.
        // Then it becomes mortal and be forced to stop at its position 
        StartCoroutine(StableCoro());
    }

    private IEnumerator StableCoro()
    {
        yield return new WaitForSeconds(ABConstants.LEVEL_IMMORTAL_TIME);
        _currentLife = _life;
    }

    protected virtual void Update()
    {
        DestroyIfOutScreen();
    }

    protected virtual void FixedUpdate()
    {
        // save velocity before physics simulation
        if(_rigidBody)
        {
            lastVelocity = _rigidBody.velocity;
        }
    }

    public virtual void Die(bool withEffect = true)
    {
        if (!ABGameWorld.Instance._isSimulation && withEffect)
        {

            _destroyEffect._shootParticles = true;
            //			ABAudioController.Instance.PlayIndependentSFX(_clips[(int)OBJECTS_SFX.DIE]);
        }

        _rigidBody.velocity = Vector2.zero;
        _spriteRenderer.color = Color.clear;
        _collider.enabled = false;

        Invoke("WaitParticlesAndDestroy", _destroyEffect._systemLifetime);
    }

    private void WaitParticlesAndDestroy()
    {

        Destroy(gameObject);
    }

    public virtual void OnCollisionEnter2D(Collision2D collision)
    {
        float damage = 0f;

        // if this object has rigidbody
        if (_rigidBody)
        {
            // relative speed
            float relativeSpeed = collision.relativeVelocity.magnitude;

            // rigidbody of the attacker, default: itself
            Rigidbody2D attackerRB = _rigidBody;

            // rigidbody of the incoming collider
            Rigidbody2D collisionRB = collision.gameObject.GetComponent<Rigidbody2D>();

            // if the incoming collider has rigidbody, choose the one which "contributes more to the relative speed" as the attacker
            if (collisionRB != null)
            {
                // VA, VB, VR
                float projectCollision = Mathf.Cos(Vector2.Angle(collision.relativeVelocity, collisionRB.velocity) * Mathf.Deg2Rad) * collisionRB.velocity.magnitude;
                float projectThis = Mathf.Cos(Vector2.Angle(collision.relativeVelocity, getRigidBody().velocity) * Mathf.Deg2Rad) * getRigidBody().velocity.magnitude;
                // if the incoming collider "contributes more to the relative speed", choose it as the attacker
                if (projectCollision > projectThis)
                {
                    attackerRB = collisionRB;
                }
            }

            // damage = relativeSpeed * massAttacker - defense
            damage = Mathf.Max(relativeSpeed * attackerRB.mass * ABConstants.DAMAGE_CONSTANT_MULTIPLIER * 0.5f - _defense, 0);

            DealDamage(damage);
        }

        /*
        // if this object has rigidbody
        if (_rigidBody)
        {
            // relative speed
            float relativeSpeed = collision.relativeVelocity.magnitude;

            // rigidbody of the incoming collider
            Rigidbody2D collisionRB = collision.gameObject.GetComponent<Rigidbody2D>();


            float totalMass = _rigidBody.mass;

            // if the incoming collider also has rigidbody
            if (collisionRB)
            {
                totalMass += collisionRB.mass;
            }

            // damage = relativeSpeed * mass * multiplier - defense
            damage = Mathf.Max(relativeSpeed * totalMass * ABConstants.DAMAGE_CONSTANT_MULTIPLIER - _defense, 0);

            // this object takes damage
            DealDamage(damage);
        }*/

        /*
        // 2D Impulse force calculation
        float impulse = 0;
        foreach (ContactPoint2D point in collision.contacts)
        {
            impulse += point.normalImpulse;
        }

        // damage = impulse * multiplier - defense
        damage = Mathf.Max(impulse * ABConstants.DAMAGE_CONSTANT_MULTIPLIER - _defense, 0);

        DealDamage(damage);
        */

        /*
        // if this object has rigidbody
        if (_rigidBody)
        {
            // kinetic energy loss amount
            float kineticLoss = 0;

            // calculate kinetic energy difference of this collider
            kineticLoss +=
                0.5f * _rigidBody.mass * (lastVelocity.magnitude - _rigidBody.velocity.magnitude); 

            // rigidbody of the incoming collider
            Rigidbody2D collisionRB = collision.gameObject.GetComponent<Rigidbody2D>();

            // if the incoming collider also has rigidbody
            if (collisionRB)
            {
                ABGameObject collisionABGameObj = collision.gameObject.GetComponent<ABGameObject>();

                // if the incoming collider is an ABGameObj
                if (collisionABGameObj)
                {
                    // calculate kinetic energy difference of incoming collider
                    kineticLoss +=
                        0.5f * collisionRB.mass * (collisionABGameObj.lastVelocity.magnitude - collisionRB.velocity.magnitude);
                }
            }

            // damage =  kinetic energy loss * multiplier - defense
            damage = Mathf.Max(0, kineticLoss * ABConstants.DAMAGE_CONSTANT_MULTIPLIER - _defense);

            // this object takes damage
            DealDamage(damage);

            // spawn the points for colliding with the bird
            if (collision.gameObject.tag == "Bird")
            {
                // give points if it only exeeds 10 and object is not destroyed
                int points = (int)Mathf.Round(damage) * 10;
                if ((points > 10) & ((getCurrentLife() - damage) > 0f))
                {
                    ScoreHud.Instance.SpawnScorePoint(points, transform.position);
                }
            }
        }
        */

        // spawn the points for colliding with the bird
        if (collision.gameObject.tag == "Bird")
        {
            // give points if it only exeeds 10 and object is not destroyed
            int points = (int)Mathf.Round(damage) * 10;
            if ((points > 10) & ((getCurrentLife() - damage) > 0f))
            {
                ScoreHud.Instance.SpawnScorePoint(points, transform.position);
            }
        }
    }

    public void DealDamage(float damage)
    {

        _currentLife -= damage;

        //if (_currentLife <= (_life / _sprites.Length * (_sprites.Length - _spriteChangedTimes + 1)))
        if (_currentLife <= _life - (_life / (_sprites.Length + 1)) * (_spriteChangedTimes + 1))
        {
            if (_spriteChangedTimes < _sprites.Length)
                _spriteRenderer.sprite = _sprites[_spriteChangedTimes];

            //if(!ABGameWorld.Instance._isSimulation)
            //_audioSource.PlayOneShot(_clips[(int)OBJECTS_SFX.DAMAGE]);

            _spriteChangedTimes++;
        }

        if (_currentLife <= 0f)
            Die();

    }

    void DestroyIfOutScreen()
    {

        if (ABGameWorld.Instance.IsObjectOutOfWorld(transform, _collider))
        {

            IsDying = true;
            Die(false);
        }
    }
    public Rigidbody2D getRigidBody()
    {
        return _rigidBody;
    }

    public float getCurrentLife()
    {
        return _currentLife;
    }
}
