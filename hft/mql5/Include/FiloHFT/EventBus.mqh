//+------------------------------------------------------------------+
//|  FiloHFT/EventBus.mqh                                            |
//|  Minimal in-process event bus. Not a queue for another thread —  |
//|  MQL5 is single-threaded per chart. This is a typed dispatch     |
//|  point so engines don't need to know about each other directly.  |
//|                                                                  |
//|  Pattern: each engine exposes a free-function OnXxx(payload).    |
//|  EventBus::Publish() calls the relevant OnXxx handlers in order. |
//|  No dynamic subscribers; wiring is done in the main EA.          |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_EVENTBUS_MQH__
#define __FILOHFT_EVENTBUS_MQH__

#include <FiloHFT/Common.mqh>

enum ENUM_FILO_EVENT {
   EVT_NONE            = 0,
   EVT_TICK            = 1,
   EVT_BAR_CLOSE       = 2,
   EVT_KL_UPDATE       = 3,
   EVT_FEATURE_UPDATE  = 4,
   EVT_SETUP_CANDIDATE = 5,
   EVT_INTENT_CREATED  = 6,
   EVT_INTENT_TRIGGERED= 7,
   EVT_FILL            = 8,
   EVT_TRADE_MODIFIED  = 9,
   EVT_TRADE_CLOSED    = 10,
   EVT_RISK_ALERT      = 11,
   EVT_SAFETY_ALERT    = 12,
   EVT_KILL_SWITCH     = 13
};

//+------------------------------------------------------------------+
//|  Generic event payload struct. Each event type uses a subset of  |
//|  fields — the consumer knows which ones are valid for its type.  |
//|  Keeping one struct avoids polymorphism in MQL5.                 |
//+------------------------------------------------------------------+
struct FiloEvent {
   ENUM_FILO_EVENT  type;
   long             ts_ms;

   // price / level fields
   double           price;
   double           aux_a;
   double           aux_b;

   // string fields
   string           tag_a;
   string           tag_b;
   string           message;

   // integer flags
   int              int_a;
   int              int_b;
   long             long_a;
};

//+------------------------------------------------------------------+
//|  The actual dispatch is wired by the main EA: it simply calls    |
//|  the next engine's handler in order inside OnTick. This file is  |
//|  kept for the types and for future expansion if we ever add      |
//|  subscribers at runtime (e.g., a debug listener).                |
//+------------------------------------------------------------------+
class EventLog {
private:
   FiloEvent m_ring[256];
   int       m_head;
   int       m_count;
public:
   void Init() { m_head = 0; m_count = 0; }
   void Push(const FiloEvent &e) {
      m_ring[m_head] = e;
      m_head = (m_head + 1) % 256;
      if(m_count < 256) m_count++;
   }
   int Size() const { return m_count; }
};

EventLog g_event_log;

#endif // __FILOHFT_EVENTBUS_MQH__
